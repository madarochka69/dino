#!/usr/bin/env python3
"""NEAT training for the Chrome Dino game in async Chromium browsers.

The game runs in real time at its normal requestAnimationFrame speed. Each
browser owns one page and one game. The injected game_agent.js reads the game
state, evaluates the genome's neural network and applies its action; there are
no hand-written game rules.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import gzip
import math
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import neat
import neat.nn
from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "neat_config.ini"
AGENT_PATH = BASE_DIR / "game_agent.js"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "neat-checkpoint.pkl"
BEST_PATH = CHECKPOINT_DIR / "best_genome.pkl"
HISTORY_PATH = CHECKPOINT_DIR / "history.csv"

# The score-only fitness rewards a degenerate policy: duck forever and jump
# when an obstacle gets close. That policy survives to the timeout but never
# learns to use ducking for mid-height pterodactyls. This fitness adds small
# rewards for the actions that actually keep the dino alive, without taking
# control away from the network. `PTERO_DUCK_BONUS` is per mid-height
# pterodactyl actually passed while ducking.
FITNESS_MODE = "ducking-v1"
DUCK_REWARD_PER_FRAME = 2.0
USELESS_DUCK_PENALTY_PER_FRAME = 0.1
BAD_JUMP_PENALTY_PER_FRAME = 0.5
USELESS_JUMP_PENALTY_PER_FRAME = 0.05
PTERO_DUCK_BONUS = 50.0
STALL_SECONDS = 8.0
EVALUATE_TIMEOUT = 8.0
# The default episode timeout is 1800 seconds (about 35k points at max speed).
# A lower 300-second run ends at roughly 5280 points; that stop is the episode
# limit, not a browser freeze. Real hangs are caught by STALL_SECONDS and
# EVALUATE_TIMEOUT above.

GAME_URL = "https://wayou.github.io/t-rex-runner/"
READY_JS = (
    "() => !!(window.Runner && window.Runner.instance_"
    " && window.Runner.instance_.tRex && window.Runner.instance_.horizon)"
)
LAUNCH_ARGS = [
    "--mute-audio",
    "--disable-extensions",
    "--disable-infobars",
    "--no-first-run",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
    "--disable-features=CalculateNativeWinOcclusion,IntensiveWakeUpThrottling,"
    "BackgroundTabLoadingFromNetwork",
]
VIEWPORT = {"width": 960, "height": 300}
# Chrome window chrome (tab strip + toolbar) around the page viewport.
WINDOW_FRAME_WIDTH = 20
WINDOW_FRAME_HEIGHT = 90


def screen_work_area() -> tuple[int, int]:
    """Usable desktop area, excluding the taskbar when possible."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(
                0x0030, 0, ctypes.byref(rect), 0  # SPI_GETWORKAREA
            ):
                return (
                    max(640, rect.right - rect.left),
                    max(360, rect.bottom - rect.top),
                )
        except Exception:
            pass
    return 1920, 1080


def tile_layout(count: int) -> list[tuple[dict, list[str]]]:
    """Viewports and Chrome window args for a compact grid of visible games."""
    if count <= 0:
        return []
    screen_w, screen_h = screen_work_area()
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / cols)
    cell_w = screen_w // cols
    cell_h = min(360, max(260, screen_h // rows))
    layout = []
    for index in range(count):
        col = index % cols
        row = index // cols
        x = min(col * cell_w, max(0, screen_w - cell_w))
        y = min(row * cell_h, max(0, screen_h - cell_h))
        viewport = {
            "width": max(320, cell_w - WINDOW_FRAME_WIDTH),
            "height": max(150, cell_h - WINDOW_FRAME_HEIGHT),
        }
        args = [
            f"--window-position={x},{y}",
            f"--window-size={cell_w},{cell_h}",
        ]
        layout.append((viewport, args))
    return layout


def load_agent_js() -> str:
    try:
        return AGENT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise SystemExit(f"Missing agent script: {AGENT_PATH}") from error


AGENT_JS = load_agent_js()


def genome_to_net(genome: neat.DefaultGenome, config: neat.Config) -> dict:
    """Serialize a genome for the small feed-forward evaluator in the page."""
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    if len(network.input_nodes) != 12 or len(network.output_nodes) != 3:
        raise ValueError(
            "neat_config.ini must define 12 inputs and 3 outputs "
            f"(got {len(network.input_nodes)} and {len(network.output_nodes)})"
        )

    nodes = []
    for node, activation, aggregation, bias, response, links in network.node_evals:
        if activation.__name__ != "tanh_activation":
            raise ValueError(f"game_agent.js only supports tanh, got {activation.__name__}")
        if aggregation.__name__ != "sum_aggregation":
            raise ValueError(
                f"game_agent.js only supports sum aggregation, got {aggregation.__name__}"
            )
        nodes.append(
            {
                "id": int(node),
                "bias": float(bias),
                "resp": float(response),
                "links": [[int(source), float(weight)] for source, weight in links],
            }
        )

    return {
        "inputs": [int(key) for key in network.input_nodes],
        "outputs": [int(key) for key in network.output_nodes],
        "nodes": nodes,
    }


class Worker:
    """One Chromium browser with one page; it plays one genome at a time."""

    def __init__(
        self,
        index: int,
        playwright,
        headless: bool,
        channel: str,
        viewport: dict | None = None,
        window_args: list[str] | None = None,
    ) -> None:
        self.index = index
        self.playwright = playwright
        self.headless = headless
        self.channel = channel
        self.viewport = viewport or VIEWPORT
        self.window_args = list(window_args or [])
        self.browser = None
        self.page = None

    async def open(self) -> None:
        self.browser = await self.playwright.chromium.launch(
            channel=self.channel,
            headless=self.headless,
            args=LAUNCH_ARGS + self.window_args,
        )
        try:
            self.page = await self.browser.new_page(viewport=self.viewport)
            self.page.set_default_timeout(30_000)
            self.page.on(
                "crash",
                lambda _page: print(
                    f"  [w{self.index:02d}] page crashed", flush=True
                ),
            )
            self.page.on(
                "pageerror",
                lambda error: print(
                    f"  [w{self.index:02d}] page error: {error}", flush=True
                ),
            )
            await self.page.goto(GAME_URL, wait_until="domcontentloaded")
            await self.page.wait_for_function(READY_JS, timeout=30_000)
            await self.page.evaluate(AGENT_JS)
            await self.page.evaluate(
                "() => { const box = document.getElementById('messageBox');"
                " if (box) box.style.display = 'none'; }"
            )
            try:
                session = await self.page.context.new_cdp_session(self.page)
                await session.send(
                    "Page.setWebLifecycleState", {"state": "active"}
                )
                await session.detach()
            except Exception:
                pass
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        browser = self.browser
        self.browser = None
        self.page = None
        if browser is not None:
            try:
                await asyncio.wait_for(browser.close(), timeout=15)
            except Exception:
                pass

    async def reset(self) -> None:
        await self.close()
        await self.open()

    async def safe_reset(self) -> None:
        print(f"  [w{self.index:02d}] restarting browser", flush=True)
        try:
            await asyncio.wait_for(self.reset(), timeout=45)
        except Exception as error:
            print(
                f"  [w{self.index:02d}] browser restart failed: {error}",
                flush=True,
            )

    async def evaluate(self, expression: str, arg=None, timeout: float | None = None):
        call = (
            self.page.evaluate(expression)
            if arg is None
            else self.page.evaluate(expression, arg)
        )
        return await asyncio.wait_for(call, timeout=timeout or EVALUATE_TIMEOUT)

    async def play(self, net: dict, timeout: float) -> dict:
        await self.evaluate("(net) => window.__dinoAgent.start(net)", net)
        started = time.monotonic()
        unlimited = timeout <= 0
        deadline = math.inf if unlimited else started + timeout
        next_report = started + 5.0
        last_progress = started
        last_state = {
            "done": False,
            "score": 0,
            "passed": 0,
            "frames": 0,
            "elapsed": 0.0,
            "crashed": False,
        }

        def stalled_result(state: dict) -> dict:
            result = dict(state)
            result.update(
                done=True,
                crashed=False,
                stalled=True,
                timed_out=False,
                failed=True,
                elapsed=time.monotonic() - started,
            )
            return result

        while True:
            await asyncio.sleep(0.25)
            try:
                state = await self.evaluate(
                    "() => window.__dinoAgent.snapshot()"
                )
            except asyncio.TimeoutError:
                print(
                    f"  [w{self.index:02d}] page unresponsive "
                    f"at score {last_state['score']}",
                    flush=True,
                )
                await self.safe_reset()
                return stalled_result(last_state)
            except Exception as error:
                print(
                    f"  [w{self.index:02d}] snapshot error: {error}",
                    flush=True,
                )
                await self.safe_reset()
                return stalled_result(last_state)

            if state["done"] or state["crashed"]:
                return state

            now = time.monotonic()
            if state["frames"] != last_state.get("frames"):
                last_progress = now
            last_state = state

            if now - last_progress >= STALL_SECONDS:
                print(
                    f"  [w{self.index:02d}] game stalled at "
                    f"score {state['score']} (no frames for "
                    f"{STALL_SECONDS:.0f}s)",
                    flush=True,
                )
                await self.safe_reset()
                return stalled_result(state)

            if not unlimited and now >= deadline:
                try:
                    await self.evaluate(
                        "() => window.__dinoAgent.stop()", timeout=5.0
                    )
                    final = await self.evaluate(
                        "() => window.__dinoAgent.snapshot()", timeout=5.0
                    )
                    final.update(
                        timed_out=True,
                        failed=False,
                        stalled=False,
                        elapsed=now - started,
                    )
                    print(
                        f"  [w{self.index:02d}] episode timeout at "
                        f"score {final['score']} ({timeout:.0f}s); "
                        "this is the training limit, not a crash "
                        "(use --episode-timeout to change it)",
                        flush=True,
                    )
                    return final
                except Exception:
                    await self.safe_reset()
                    return stalled_result(last_state)

            if now >= next_report:
                print(
                    f"  [w{self.index:02d}] score {state['score']}",
                    flush=True,
                )
                next_report += 5.0


async def start_workers(
    playwright,
    count: int,
    visible: bool,
    channel: str,
    visible_count: int = 0,
) -> list[Worker]:
    visible_count = max(0, min(visible_count, count)) if visible else 0
    layout = tile_layout(visible_count)
    workers = []
    try:
        for index in range(count):
            viewport = layout[index][0] if index < visible_count else None
            window_args = layout[index][1] if index < visible_count else None
            worker = Worker(
                index,
                playwright,
                headless=index >= visible_count,
                channel=channel,
                viewport=viewport,
                window_args=window_args,
            )
            await worker.open()
            workers.append(worker)
            print(f"  browser {index + 1}/{count} ready", flush=True)
    except Exception:
        for worker in workers:
            await worker.close()
        raise
    return workers


def shaped_fitness(result: dict) -> float:
    """Score from the episode plus the ducking behaviour we want to learn."""
    useful_duck = int(result.get("useful_duck_frames", 0))
    useless_duck = int(result.get("useless_duck_frames", 0))
    bad_jump = int(result.get("bad_jump_frames", 0))
    useless_jump = int(result.get("useless_jump_frames", 0))
    pteros_ducked = int(result.get("pterodactyls_ducked", 0))
    return (
        float(result.get("score", 0))
        + DUCK_REWARD_PER_FRAME * useful_duck
        - USELESS_DUCK_PENALTY_PER_FRAME * useless_duck
        - BAD_JUMP_PENALTY_PER_FRAME * bad_jump
        - USELESS_JUMP_PENALTY_PER_FRAME * useless_jump
        + PTERO_DUCK_BONUS * pteros_ducked
    )


def empty_episode() -> dict:
    return {
        "score": 0,
        "passed": 0,
        "frames": 0,
        "elapsed": 0.0,
        "crashed": True,
        "stalled": False,
        "timed_out": False,
        "failed": True,
    }


async def evaluate_population(
    workers: list[Worker],
    config: neat.Config,
    genomes: list[tuple[int, neat.DefaultGenome]],
    timeout: float,
    episodes: int,
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for item in genomes:
        queue.put_nowait(item)

    async def run(worker: Worker) -> None:
        while True:
            try:
                genome_id, genome = queue.get_nowait()
            except asyncio.QueueEmpty:
                print(
                    f"  [w{worker.index:02d}] waiting for the generation "
                    "to finish (game stopped, browser is not frozen)",
                    flush=True,
                )
                return

            net = genome_to_net(genome, config)
            results = []
            for episode in range(episodes):
                try:
                    result = await worker.play(net, timeout)
                except Exception as error:
                    print(
                        f"  [w{worker.index:02d}] browser error on "
                        f"episode {episode + 1}/{episodes}: {error}",
                        flush=True,
                    )
                    await worker.safe_reset()
                    result = empty_episode()
                results.append(result)

            scores = [float(result.get("score", 0)) for result in results]
            shaped = [shaped_fitness(result) for result in results]
            raw_score = sum(scores) / len(scores)
            fitness = sum(shaped) / len(shaped)

            genome.fitness = fitness
            genome.raw_score = raw_score
            genome.fitness_mode = FITNESS_MODE
            genome.episode_scores = scores
            genome.passed = sum(int(result.get("passed", 0)) for result in results)
            genome.frames = sum(int(result.get("frames", 0)) for result in results)
            genome.duck_frames = sum(
                int(result.get("duck_frames", 0)) for result in results
            )
            genome.useful_duck_frames = sum(
                int(result.get("useful_duck_frames", 0)) for result in results
            )
            genome.useless_duck_frames = sum(
                int(result.get("useless_duck_frames", 0)) for result in results
            )
            genome.bad_jump_frames = sum(
                int(result.get("bad_jump_frames", 0)) for result in results
            )
            genome.useless_jump_frames = sum(
                int(result.get("useless_jump_frames", 0)) for result in results
            )
            genome.pterodactyls_passed = sum(
                int(result.get("pterodactyls_passed", 0)) for result in results
            )
            genome.pterodactyls_duckable = sum(
                int(result.get("pterodactyls_duckable", 0)) for result in results
            )
            genome.pterodactyls_ducked = sum(
                int(result.get("pterodactyls_ducked", 0)) for result in results
            )
            genome.episodes = len(results)
            genome.timed_out_episodes = sum(
                1 for result in results if result.get("timed_out")
            )
            genome.stalled_episodes = sum(
                1 for result in results if result.get("stalled")
            )
            genome.failed_episodes = sum(
                1 for result in results if result.get("failed")
            )

            duck_pct = (
                100.0 * genome.duck_frames / genome.frames
                if genome.frames
                else 0.0
            )
            flags = []
            if genome.timed_out_episodes:
                flags.append(f"TIMEOUT {genome.timed_out_episodes}/{episodes}")
            if genome.stalled_episodes:
                flags.append(f"STALLED {genome.stalled_episodes}/{episodes}")
            if genome.failed_episodes:
                flags.append(f"FAILED {genome.failed_episodes}/{episodes}")
            print(
                f"  [w{worker.index:02d}] genome {genome_id:3d} | "
                f"score {raw_score:6.0f} ({', '.join(f'{value:.0f}' for value in scores)}) | "
                f"fit {fitness:7.1f} | "
                f"passed {genome.passed:3d} | duck {duck_pct:3.0f}% | "
                f"ptero {genome.pterodactyls_ducked}/"
                f"{genome.pterodactyls_duckable} of "
                f"{genome.pterodactyls_passed} | "
                f"{sum(float(result.get('elapsed', 0.0)) for result in results):6.1f}s"
                + (" | " + " | ".join(flags) if flags else ""),
                flush=True,
            )

    await asyncio.gather(*(run(worker) for worker in workers))


def save_checkpoint(population: neat.Population, config: neat.Config) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=5) as handle:
        pickle.dump(
            (
                population.generation,
                config,
                population.population,
                population.species,
                random.getstate(),
            ),
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, CHECKPOINT_PATH)
    print(f"  checkpoint -> {CHECKPOINT_PATH} (generation {population.generation})")


def save_best(genome: neat.DefaultGenome) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    genome.fitness_mode = FITNESS_MODE
    if getattr(genome, "raw_score", None) is None:
        genome.raw_score = genome.fitness
    temporary = BEST_PATH.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(genome, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, BEST_PATH)
    print(
        f"  best genome -> {BEST_PATH} "
        f"(raw {genome.raw_score:.0f}, fitness {genome.fitness:.0f}, "
        f"passed {getattr(genome, 'passed', 0)})"
    )


def load_best() -> neat.DefaultGenome | None:
    if not BEST_PATH.exists():
        return None
    try:
        with open(BEST_PATH, "rb") as handle:
            return pickle.load(handle)
    except Exception as error:
        print(f"Could not read {BEST_PATH}: {error}")
        return None


def archive_legacy_best() -> None:
    """Keep a copy of a best genome from an older fitness version."""
    if not BEST_PATH.exists():
        return
    best = load_best()
    if best is not None and getattr(best, "fitness_mode", None) == FITNESS_MODE:
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = CHECKPOINT_DIR / f"best_genome_legacy_{stamp}.pkl"
    try:
        shutil.copyfile(BEST_PATH, target)
        print(f"  legacy best archived -> {target}", flush=True)
    except OSError as error:
        print(f"  could not archive legacy best: {error}", flush=True)


def load_current_best() -> neat.DefaultGenome | None:
    best = load_best()
    if best is None:
        return None
    if getattr(best, "fitness_mode", None) != FITNESS_MODE:
        archive_legacy_best()
        print(
            "  saved best uses the old score-only fitness; "
            "it will not be reinserted into the ducking run.",
            flush=True,
        )
        return None
    return best


def keep_best(
    population: dict[int, neat.DefaultGenome],
    best: neat.DefaultGenome | None,
    config: neat.Config,
) -> None:
    """Keep the all-time best in the breeding population.

    NEAT only protects elites inside their own species. If that species
    stagnates or its fitness drops, the best genome can be lost even though
    it is saved in best_genome.pkl. Re-insert it (with a fresh evaluation)
    so training cannot regress below the saved result.
    """
    if best is None or best.fitness is None or best.key in population:
        return
    if getattr(best, "fitness_mode", None) != FITNESS_MODE:
        return

    clone = copy.deepcopy(best)
    clone.fitness = None
    clone.raw_score = None
    clone.passed = int(getattr(best, "passed", 0))
    if clone.key in population:
        clone.key = max(population, default=0) + 1
    population[clone.key] = clone

    while len(population) > config.pop_size:
        victim = min(
            (genome for genome in population.values() if genome.key != clone.key),
            key=lambda genome: (
                genome.fitness is not None,
                genome.fitness if genome.fitness is not None else -1.0,
                genome.key,
            ),
        )
        del population[victim.key]


HISTORY_FIELDS = [
    "generation",
    "best_key",
    "best_score",
    "best_fitness",
    "best_passed",
    "best_duck_pct",
    "best_ptero_ducked",
    "best_ptero_duckable",
    "best_ptero_passed",
    "mean_score",
    "mean_fitness",
    "evaluated",
    "species",
    "global_best_fitness",
    "global_best_score",
    "seconds",
    "time",
]


def save_history(
    generation: int,
    best: neat.DefaultGenome,
    mean_raw: float,
    mean_fitness: float,
    species_count: int,
    evaluated: int,
    global_best: neat.DefaultGenome | None,
    elapsed: float,
) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    new_file = not HISTORY_PATH.exists()
    if not new_file:
        with open(HISTORY_PATH, "r", newline="", encoding="utf-8") as handle:
            header = handle.readline().strip()
        if header != ",".join(HISTORY_FIELDS):
            stamp = time.strftime("%Y%m%d-%H%M%S")
            legacy = CHECKPOINT_DIR / f"history_legacy_{stamp}.csv"
            HISTORY_PATH.rename(legacy)
            print(f"  history format changed; old log -> {legacy}", flush=True)
            new_file = True

    best_raw = float(getattr(best, "raw_score", best.fitness))
    best_frames = max(1, int(getattr(best, "frames", 0)))
    best_duck_pct = 100.0 * int(getattr(best, "duck_frames", 0)) / best_frames
    global_best_raw = (
        float(getattr(global_best, "raw_score", global_best.fitness))
        if global_best
        else None
    )

    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(HISTORY_FIELDS)
        writer.writerow(
            [
                generation,
                best.key,
                f"{best_raw:.0f}",
                f"{best.fitness:.1f}",
                int(getattr(best, "passed", 0)),
                f"{best_duck_pct:.1f}",
                int(getattr(best, "pterodactyls_ducked", 0)),
                int(getattr(best, "pterodactyls_duckable", 0)),
                int(getattr(best, "pterodactyls_passed", 0)),
                f"{mean_raw:.1f}",
                f"{mean_fitness:.1f}",
                evaluated,
                species_count,
                f"{global_best.fitness:.1f}" if global_best else "",
                f"{global_best_raw:.0f}" if global_best_raw is not None else "",
                f"{elapsed:.1f}",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ]
        )


async def train(args: argparse.Namespace, config: neat.Config) -> None:
    if args.resume and CHECKPOINT_PATH.exists():
        population = neat.Checkpointer.restore_checkpoint(str(CHECKPOINT_PATH), config)
        print(
            f"Resumed {CHECKPOINT_PATH} at generation {population.generation}.",
            flush=True,
        )
        global_best = load_current_best()
        if global_best is None and BEST_PATH.exists():
            print(
                "  saved best is from the old score-only fitness and will "
                "not be reinserted; the population is re-evaluated with "
                f"{FITNESS_MODE} fitness.",
                flush=True,
            )
        best_score = (
            float(global_best.fitness)
            if global_best is not None and global_best.fitness is not None
            else -math.inf
        )
    else:
        population = neat.Population(config)
        load_current_best()  # archive a legacy best so it is not replayed
        global_best = None
        best_score = -math.inf
        print("Starting a new population from scratch.", flush=True)

    if population.generation >= args.generations:
        print(
            f"Nothing to do: generation {population.generation} "
            f">= limit {args.generations}."
        )
        return

    if args.resume and global_best is not None:
        if global_best.key not in population.population:
            keep_best(population.population, global_best, config)
            population.species.speciate(
                config, population.population, population.generation
            )
            print(
                f"Reinserted saved best genome {global_best.key} "
                f"(score {global_best.fitness:.0f}) into the population.",
                flush=True,
            )

    worker_count = max(1, min(args.games, config.pop_size))
    visible_count = (
        max(1, min(args.visible_games, worker_count)) if args.visible else 0
    )
    print(
        f"Workers: {worker_count} {args.browser} browsers "
        f"({visible_count} visible, {worker_count - visible_count} headless), "
        f"population: {config.pop_size}, generations: {args.generations}.",
        flush=True,
    )
    if visible_count:
        print(
            "  Showing the first "
            f"{visible_count} games in a grid; all {worker_count} still train. "
            "Use --visible-games 1 for one large window.",
            flush=True,
        )

    async with async_playwright() as playwright:
        workers = await start_workers(
            playwright,
            worker_count,
            args.visible,
            args.browser,
            visible_count,
        )
        try:
            while population.generation < args.generations:
                generation = population.generation
                started = time.perf_counter()
                print(
                    f"\nGeneration {generation + 1}/{args.generations}",
                    flush=True,
                )

                await evaluate_population(
                    workers,
                    config,
                    list(population.population.items()),
                    args.episode_timeout,
                    args.episodes,
                )

                best = max(population.population.values(), key=lambda item: item.fitness)
                mean_fitness = sum(
                    genome.fitness for genome in population.population.values()
                ) / len(population.population)
                mean_raw = sum(
                    float(getattr(genome, "raw_score", 0))
                    for genome in population.population.values()
                ) / len(population.population)
                best_raw = float(getattr(best, "raw_score", best.fitness))
                elapsed = time.perf_counter() - started
                print(
                    f"Generation {generation + 1} | "
                    f"best score {best_raw:.0f} | best fit {best.fitness:.0f} | "
                    f"mean score {mean_raw:.0f} | mean fit {mean_fitness:.0f} | "
                    f"species {len(population.species.species)} | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

                if global_best is None or best.fitness > best_score:
                    best_score = float(best.fitness)
                    global_best = copy.deepcopy(best)
                    save_best(copy.deepcopy(global_best))

                save_history(
                    generation + 1,
                    best,
                    mean_raw,
                    mean_fitness,
                    len(population.species.species),
                    sum(
                        1
                        for genome in population.population.values()
                        if genome.fitness is not None
                    ),
                    global_best,
                    elapsed,
                )

                if not config.no_fitness_termination:
                    if best_score >= config.fitness_threshold:
                        print("Fitness threshold reached; stopping.", flush=True)
                        break

                population.population = population.reproduction.reproduce(
                    config,
                    population.species,
                    config.pop_size,
                    population.generation,
                )
                keep_best(population.population, global_best, config)

                if not population.species.species:
                    print("All species extinct; resetting the population.", flush=True)
                    if not config.reset_on_extinction:
                        raise neat.CompleteExtinctionException()
                    population.population = population.reproduction.create_new(
                        config.genome_type,
                        config.genome_config,
                        config.pop_size,
                    )
                    keep_best(population.population, global_best, config)

                population.species.speciate(
                    config, population.population, population.generation
                )
                population.generation += 1
                save_checkpoint(population, config)
        finally:
            for worker in workers:
                await worker.close()

    final_raw = (
        f", raw score {float(getattr(global_best, 'raw_score', 0)):.0f}"
        if global_best is not None
        else ""
    )
    print(f"Training finished. Best fitness: {best_score:.0f}{final_raw}", flush=True)


async def watch_best(args: argparse.Namespace, config: neat.Config) -> None:
    """Replay the saved best genome in one large visible window."""
    best = load_best()
    if best is None or best.fitness is None:
        raise SystemExit(f"No saved best genome at {BEST_PATH}")
    if getattr(best, "fitness_mode", None) != FITNESS_MODE:
        print(
            "  warning: this best genome was trained with score-only "
            "fitness; it may still duck constantly. Train a new run to get "
            "the ducking fitness.",
            flush=True,
        )

    print(
        f"Watching best genome {best.key} (fitness {best.fitness:.0f}). "
        "Ctrl+C to stop.",
        flush=True,
    )
    async with async_playwright() as playwright:
        worker = Worker(
            0,
            playwright,
            headless=False,
            channel=args.browser,
            viewport={"width": 1280, "height": 400},
        )
        await worker.open()
        try:
            await worker.page.bring_to_front()
            while True:
                current = load_best() or best
                best = current
                net = genome_to_net(current, config)
                result = await worker.play(net, args.episode_timeout)
                status = "timeout" if result.get("timed_out") else (
                    "stalled" if result.get("stalled") else "crash"
                )
                print(
                    f"  best {current.key} episode: score {int(result['score'])}, "
                    f"passed {int(result['passed'])}, "
                    f"{result['elapsed']:.1f}s, {status}",
                    flush=True,
                )
                await asyncio.sleep(1.0)
        finally:
            await worker.close()


async def check(args: argparse.Namespace, config: neat.Config) -> None:
    population = neat.Population(config)
    _, genome = next(iter(population.population.items()))
    net = genome_to_net(genome, config)
    timeout = (
        30.0
        if args.episode_timeout <= 0
        else min(args.episode_timeout, 30.0)
    )

    async with async_playwright() as playwright:
        workers = await start_workers(
            playwright, 1, args.visible, args.browser, 1
        )
        try:
            print("Check: one random genome, one real-time episode.", flush=True)
            result = await workers[0].play(net, timeout)
            print(
                "Check OK: "
                f"score {int(result['score'])}, "
                f"passed {int(result['passed'])}, "
                f"frames {int(result['frames'])}, "
                f"{result['elapsed']:.1f}s",
                flush=True,
            )
        finally:
            for worker in workers:
                await worker.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEAT training for Chrome Dino in async Chromium browsers."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run one random genome in one browser and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"continue from {CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--watch-best",
        action="store_true",
        help="replay checkpoints/best_genome.pkl in one large visible window",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=15,
        help="parallel Chromium browsers (default: 15)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="total number of generations (default: 100)",
    )
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=1800.0,
        help=(
            "maximum seconds per episode; 0 = no limit "
            "(default: 1800, about 35k points at max speed)"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="episodes per genome, averaged to reduce luck (default: 2)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="show browser windows (only the first --visible-games are visible)",
    )
    parser.add_argument(
        "--visible-games",
        type=int,
        default=4,
        help="number of games to show when --visible is used (default: 4)",
    )
    parser.add_argument(
        "--browser",
        choices=("chromium", "chrome", "msedge"),
        default="chromium",
        help="browser channel: Playwright chromium, installed Chrome or Edge",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"NEAT config file (default: {CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.visible and not args.watch_best and not args.check:
        print(
            "Browsers are headless. Pass --visible to watch the games.",
            flush=True,
        )

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(args.config),
    )
    args.games = max(1, args.games)
    args.generations = max(1, args.generations)
    args.episodes = max(1, args.episodes)
    args.episode_timeout = max(0.0, args.episode_timeout)
    args.visible_games = max(1, min(args.visible_games, args.games))

    try:
        if args.check:
            asyncio.run(check(args, config))
        elif args.watch_best:
            asyncio.run(watch_best(args, config))
        else:
            asyncio.run(train(args, config))
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    except Exception as error:
        print(f"Error: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
