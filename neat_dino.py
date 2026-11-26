#!/usr/bin/env python3
"""
NEAT-обучение для Chrome Dino (T-Rex Runner).
Запускает параллельные страницы Chromium через Playwright,
каждая страница оценивает геном в отдельном эпизоде.

Наблюдения (6 входов NEAT):
  1. distance / 600  — расстояние до ближайшего препятствия (0..1)
  2. obs_width / 50  — ширина препятствия (0..1)
  3. obs_height / 50 — высота препятствия (0..1)
  4. obs_y / 150     — Y-позиция препятствия (0..1, для птеродактилей)
  5. speed / 13      — текущая скорость игры (0..1)
  6. is_jumping      — 1 если динозавр в прыжке, иначе 0

Выход NEAT (1 нейрон, tanh):
  > 0 → прыжок (вызов dino.startJump())
  ≤ 0 → не прыгать

Фитнес: дистанция пробега (runner.distanceRan)
"""

import asyncio
import json
import os
import random
import sys
import time

import neat
import numpy as np
from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
# кодировка для Windows
import sys
if sys.stdout.encoding.lower() in ('cp1251', 'cp866'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GAME_URL = "https://wayou.github.io/t-rex-runner/"
NUM_WORKERS = 20  # количество параллельных страниц Chromium
MAX_STEPS = 3000  # макс. кадров на эпизод (~50 сек при 60fps)
MAX_GENERATIONS = 200
CHECKPOINT_FREQ = 5  # сохранять каждые N поколений
EPISODE_TIMEOUT = 180  # секунд на один эпизод (3 минуты макс)
HEADLESS = True  # headless-режим (False для отладки)

# ---------------------------------------------------------------------------
# JS-функция для выполнения одного эпизода внутри браузера
# ---------------------------------------------------------------------------
RUN_EPISODE_JS = """
async (params) => {
    const { weights, maxSteps } = params;

    // ---------- feedforward нейросеть ----------
    function evaluateNetwork(inputs) {
        const values = {};
        // входные нейроны
        for (let i = 0; i < weights.input_ids.length; i++) {
            values[weights.input_ids[i]] = inputs[i];
        }
        // скрытые/выходные нейроны (сортируем по ID для топологической упорядоченности)
        const nodeIds = Object.keys(weights.nodes).map(Number).sort((a, b) => a - b);
        for (const nodeId of nodeIds) {
            const node = weights.nodes[nodeId];
            let sum = node.bias;
            for (const [fromId, weight] of node.inputs) {
                if (fromId in values) {
                    sum += values[fromId] * weight;
                }
            }
            values[nodeId] = Math.tanh(node.response * sum);
        }
        return weights.output_ids.map(id => values[id] || 0);
    }

    // ---------- ожидание Runner ----------
    const waitForRunner = () => new Promise(resolve => {
        if (window.Runner && window.Runner.instance_) {
            resolve(window.Runner.instance_);
            return;
        }
        const check = setInterval(() => {
            if (window.Runner && window.Runner.instance_) {
                clearInterval(check);
                resolve(window.Runner.instance_);
            }
        }, 50);
        setTimeout(() => { clearInterval(check); resolve(null); }, 5000);
    });

    const runner = await waitForRunner();
    if (!runner) return { fitness: 1, steps: 0, error: 'NoRunner' };

    // ---------- фокус на canvas ----------
    const canvas = document.querySelector('canvas');
    if (canvas) canvas.focus();

    // ---------- запуск / рестарт игры ----------
    if (runner.crashed) {
        // предыдущий эпизод закончился — просто рестартуем
        runner.restart();
        await new Promise(r => setTimeout(r, 200));
    } else if (!runner.playing) {
        // игра ещё не началась — эмулируем пробел
        document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, bubbles: true }));
        document.dispatchEvent(new KeyboardEvent('keyup', { keyCode: 32, bubbles: true }));
        await new Promise(r => setTimeout(r, 300));
    }

    if (runner.crashed || !runner.playing) {
        // не удалось запустить
        return { fitness: 1, steps: 0, error: 'StartFailed' };
    }

    // ---------- игровой цикл ----------
    return new Promise(resolve => {
        let steps = 0;
        const dino = runner.tRex;

        function gameLoop() {
            steps++;

            // проверка окончания
            if (runner.crashed || steps >= maxSteps) {
                resolve({
                    fitness: Math.max(1, Math.floor(runner.distanceRan || 0)),
                    steps: steps,
                });
                return;
            }

            // ---- чтение состояния игры ----
            const obstacles = runner.horizon ? runner.horizon.obstacles : [];
            const speed = runner.currentSpeed || 6;

            let dist = 600;
            let obsWidth = 0;
            let obsHeight = 0;
            let obsY = 0;

            if (obstacles.length > 0) {
                const obs = obstacles[0];
                dist = obs.xPos - (dino ? dino.xPos : 50);
                obsWidth = obs.width || (obs.typeConfig ? obs.typeConfig.width : 17);
                obsHeight = obs.typeConfig ? obs.typeConfig.height : 35;
                obsY = obs.yPos;
            }

            // ---- нормализованные входы ----
            const inputs = [
                Math.max(0, Math.min(1, dist / 600)),
                Math.min(1, obsWidth / 50),
                Math.min(1, obsHeight / 50),
                obsY / 150,
                Math.min(1, speed / 13),
                (dino && dino.jumping) ? 1.0 : 0.0,
            ];

            // ---- прогон нейросети ----
            let output;
            try {
                output = evaluateNetwork(inputs);
            } catch (e) {
                output = [0];
            }

            // ---- действие ----
            if (output[0] > 0.0 && dino && !dino.jumping && !dino.ducking) {
                try { dino.startJump(speed); } catch (e) {}
            }

            // ---- следующий кадр ----
            requestAnimationFrame(gameLoop);
        }

        requestAnimationFrame(gameLoop);
    });
}
"""


# ---------------------------------------------------------------------------
# Сериализация генома NEAT в JSON для браузера
# ---------------------------------------------------------------------------
def genome_to_serializable(genome, config):
    """
    Превращает геном NEAT в простой dict, который можно передать в JS.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    nodes = {}
    input_ids = [int(i) for i in sorted(net.input_nodes)]
    output_ids = [int(i) for i in sorted(net.output_nodes)]

    # net.node_evals: (node_id, activation_fn, aggregation_fn, bias, response, links)
    for node_id, _activation_fn, _aggregation_fn, bias, response, links in net.node_evals:
        node_inputs = [[int(from_id), float(w)] for from_id, w in links]
        nodes[str(node_id)] = {
            "bias": float(bias),
            "response": float(response),
            "inputs": node_inputs,
        }

    return {
        "input_ids": input_ids,
        "output_ids": output_ids,
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Запуск одного эпизода на странице
# ---------------------------------------------------------------------------
async def run_single_episode(page, serialized_genome, max_steps=MAX_STEPS):
    """Запускает один эпизод игры на странице с переданным геномом."""
    try:
        result = await asyncio.wait_for(
            page.evaluate(RUN_EPISODE_JS, {
                "weights": serialized_genome,
                "maxSteps": max_steps,
            }),
            timeout=EPISODE_TIMEOUT,
        )
        return result.get("fitness", 1)
    except asyncio.TimeoutError:
        print("    ⏱ Таймаут эпизода!")
        return 1
    except Exception as exc:
        print(f"    ⚠ Ошибка: {exc}")
        return 1


# ---------------------------------------------------------------------------
# Worker — оценивает пачку геномов на одной странице
# ---------------------------------------------------------------------------
async def worker_evaluate(browser, genome_chunk, config):
    """Оценивает chunk геномов последовательно на одной странице."""
    context = await browser.new_context(viewport={"width": 800, "height": 600})
    page = await context.new_page()

    try:
        await page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        for genome_id, genome in genome_chunk:
            serialized = genome_to_serializable(genome, config)
            fitness = await run_single_episode(page, serialized)

            # если фитнес подозрительно низкий — перезагружаем страницу
            if fitness < 10:
                try:
                    await page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(2000)
                except Exception:
                    pass

            genome.fitness = max(1, fitness)
            print(f"    Genome {genome_id}: fitness={genome.fitness:.0f}")

    finally:
        await context.close()


# ---------------------------------------------------------------------------
# NEAT evaluation — асинхронный вызов из NEAT
# ---------------------------------------------------------------------------
async def eval_genomes_async(genomes, config):
    """Основная асинхронная оценка всех геномов поколения."""
    genome_list = list(genomes)  # (genome_id, genome)

    # тасуем для равномерной нагрузки
    random.shuffle(genome_list)

    # делим на chunks по числу воркеров
    n_workers = min(NUM_WORKERS, len(genome_list))
    chunks = [[] for _ in range(n_workers)]
    for i, g in enumerate(genome_list):
        chunks[i % n_workers].append(g)

    print(f"  Оцениваю {len(genome_list)} геномов на {n_workers} воркерах...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chromium",
            headless=HEADLESS,
            args=[
                "--disable-gpu",
                "--disable-extensions",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        tasks = [worker_evaluate(browser, chunk, config) for chunk in chunks if chunk]
        await asyncio.gather(*tasks)
        await browser.close()


def eval_genomes(genomes, config):
    """Синхронная обёртка для NEAT."""
    asyncio.run(eval_genomes_async(genomes, config))


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------
def run_neat(resume_from=None):
    """Запускает NEAT-обучение."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neat_config_legacy.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if resume_from and os.path.exists(resume_from):
        print(f"[*] Загружаю контрольную точку: {resume_from}")
        pop = neat.Checkpointer.restore_checkpoint(resume_from)
    else:
        pop = neat.Population(config)
        print("[*] Создана новая популяция")

    # репортёры
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=CHECKPOINT_FREQ,
        filename_prefix="neat-checkpoint-",
    ))

    print(f"\n{'='*60}")
    print("Запуск NEAT-обучения Chrome Dino")
    print(f"Параллельных воркеров: {NUM_WORKERS}")
    print(f"Макс. поколений: {MAX_GENERATIONS}")
    print(f"Макс. шагов на эпизод: {MAX_STEPS}")
    print(f"{'='*60}\n")

    winner = pop.run(eval_genomes, MAX_GENERATIONS)

    print(f"\n{'='*60}")
    print("*** ОБУЧЕНИЕ ЗАВЕРШЕНО! ***")
    print(f"{'='*60}")
    print(f"Лучший геном: {winner}")
    print(f"Фитнес: {winner.fitness}")

    # сохраняем лучший геном
    import pickle
    with open("winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("[-] Лучший геном сохранён в winner_genome.pkl")

    # показываем итоговую сеть
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    print(f"Сеть лучшего генома: {net}")
    print()

    return winner


def main():
    global NUM_WORKERS, MAX_GENERATIONS, HEADLESS
    import argparse

    parser = argparse.ArgumentParser(description="NEAT-обучение Chrome Dino")
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Путь к checkpoint для возобновления (neat-checkpoint-N)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=NUM_WORKERS,
        help=f"Количество параллельных страниц (по умолч. {NUM_WORKERS})",
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=MAX_GENERATIONS,
        help=f"Максимум поколений (по умолч. {MAX_GENERATIONS})",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Показать окно Chromium (по умолч. headless)",
    )

    args = parser.parse_args()

    NUM_WORKERS = args.workers
    MAX_GENERATIONS = args.generations
    HEADLESS = not args.visible

    run_neat(resume_from=args.resume)


if __name__ == "__main__":
    main()