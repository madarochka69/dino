// In-page agent for the T-Rex Runner.
// The game runs at its normal requestAnimationFrame speed; this file only
// reads observations, evaluates the NEAT network and applies its action.
(() => {
  "use strict";

  const runner = window.Runner && window.Runner.instance_;
  if (!runner) throw new Error("Runner is not ready");
  if (window.__dinoAgent) return;

  const clamp = (value) => Math.max(0, Math.min(1, value));

  // A duck only counts as useful when the pterodactyl is close enough that
  // ducking is the actual answer. The old 400 px window rewarded ducking
  // during the whole approach, even when the network then jumped over it.
  const DUCK_REWARD_DISTANCE = 200;

  // The two nearest obstacles that are not completely behind the dino.
  const obstaclesAhead = () => {
    const trex = runner.tRex;
    const found = [];
    for (const obstacle of runner.horizon.obstacles) {
      if (obstacle.xPos + obstacle.width >= trex.xPos) {
        found.push(obstacle);
        if (found.length === 2) break;
      }
    }
    return found;
  };

  const agent = {
    net: null,
    active: false,
    done: false,
    started: 0,
    frames: 0,
    passed: 0,
    last_action: "idle",
    action_counts: { jump: 0, duck: 0, idle: 0 },
    useful_duck_frames: 0,
    useless_duck_frames: 0,
    bad_jump_frames: 0,
    useless_jump_frames: 0,
    pterodactyls_passed: 0,
    pterodactyls_duckable: 0,
    pterodactyls_ducked: 0,

    score() {
      const raw = runner.distanceRan || 0;
      const meter = runner.distanceMeter;
      return meter && meter.getActualDistance
        ? meter.getActualDistance(Math.ceil(raw))
        : Math.round(raw);
    },

    snapshot() {
      return {
        done: this.done,
        score: this.score(),
        passed: this.passed,
        frames: this.frames,
        elapsed: this.started ? (performance.now() - this.started) / 1000 : 0,
        crashed: !!runner.crashed,
        last_action: this.last_action,
        jump_frames: this.action_counts.jump,
        duck_frames: this.action_counts.duck,
        idle_frames: this.action_counts.idle,
        useful_duck_frames: this.useful_duck_frames,
        useless_duck_frames: this.useless_duck_frames,
        bad_jump_frames: this.bad_jump_frames,
        useless_jump_frames: this.useless_jump_frames,
        pterodactyls_passed: this.pterodactyls_passed,
        pterodactyls_duckable: this.pterodactyls_duckable,
        pterodactyls_ducked: this.pterodactyls_ducked,
      };
    },

    // 12 observations. Keep the order in sync with num_inputs in neat_config.ini.
    sense() {
      const trex = runner.tRex;
      const [first, second] = obstaclesAhead();
      const type = first ? first.typeConfig.type : "";
      return [
        first ? 1 - clamp((first.xPos - trex.xPos) / 400) : 0, // distance
        first ? clamp(first.width / 100) : 0,                 // width
        first ? clamp(first.yPos / 150) : 0,                  // vertical position
        first ? clamp(first.typeConfig.height / 60) : 0,      // height
        type === "CACTUS_SMALL" ? 1 : 0,                      // type
        type === "CACTUS_LARGE" ? 1 : 0,
        type === "PTERODACTYL" ? 1 : 0,
        second ? 1 - clamp((second.xPos - trex.xPos) / 400) : 0,
        first ? clamp(first.gap / 400) : 0,                   // gap after first
        clamp((trex.groundYPos - trex.yPos) / 100),           // jump height
        trex.jumping ? 1 : 0,
        clamp(runner.currentSpeed / 13),
      ];
    },

    // The config fixes activation to tanh and aggregation to sum.
    // neat-python's tanh_activation is tanh(2.5 * z) with z clamped to
    // [-60, 60]; use the same function here so the page and Python agree.
    activate(inputs) {
      const net = this.net;
      const values = {};
      for (let i = 0; i < net.inputs.length; i++) {
        values[net.inputs[i]] = inputs[i];
      }
      for (const node of net.nodes) {
        let sum = 0;
        for (const [source, weight] of node.links) {
          sum += (values[source] || 0) * weight;
        }
        const z = 2.5 * (node.bias + node.resp * sum);
        values[node.id] = Math.tanh(Math.max(-60, Math.min(60, z)));
      }
      return net.outputs.map((id) => values[id]);
    },

    step() {
      if (!this.active) return;
      if (runner.crashed) {
        this.finish();
        return;
      }

      this.frames++;
      const [jump, duck, idle] = this.activate(this.sense());
      const trex = runner.tRex;
      // The game moved the obstacles during the update before this action.
      // Track the pose that was actually active during that update, otherwise
      // a pterodactyl passed while ducking is lost when the action switches
      // to idle/jump on the same frame.
      const wasDucking = trex.ducking;

      let action = "idle";
      if (jump >= duck && jump >= idle) action = "jump";
      else if (duck >= idle) action = "duck";
      this.last_action = action;
      this.action_counts[action]++;

      // Reward shaping telemetry. This does not control the dino: it only
      // records whether the network used the right action for the obstacle in
      // front of it. Cacti and low pterodactyls (yPos 100) are jumped; middle
      // pterodactyls (yPos 75) are ducked; high ones (yPos 50) are run under.
      const first = obstaclesAhead()[0];
      const near = !!first &&
        first.xPos - trex.xPos <= DUCK_REWARD_DISTANCE &&
        first.xPos + first.width >= trex.xPos;
      const ptero = near && first.typeConfig.type === "PTERODACTYL" ? first : null;
      const duckable = !!ptero && ptero.yPos >= 60 && ptero.yPos < 90;
      const safeHigh = !!ptero && ptero.yPos <= 60;
      if (action === "duck") {
        // Ducking while airborne is a speed drop, not a duck under the
        // pterodactyl; do not pay the useful-duck reward for it.
        if (duckable && !trex.jumping) this.useful_duck_frames++;
        else this.useless_duck_frames++;
      } else if (action === "jump") {
        if (duckable) this.bad_jump_frames++;
        else if (safeHigh || !near) this.useless_jump_frames++;
      }

      // Mimic the real controls: jump is "hold jump", idle is "release",
      // duck is "duck / speed-drop". Holding jump keeps the jump going.
      // A speed drop is a one-shot action: setSpeedDrop() resets jumpVelocity
      // to 1, so calling it every frame would hold the dino in a slow float
      // instead of letting the native 3x drop accelerate. Releasing duck also
      // clears speedDrop, exactly like the game's duck keyup handler.
      if (action === "jump") {
        // Jump is pressed, so duck is released. Native onKeyUp for duck
        // clears speedDrop; do the same before starting a jump.
        if (trex.speedDrop) trex.speedDrop = false;
        if (trex.ducking) trex.setDuck(false);
        if (!trex.jumping) trex.startJump(runner.currentSpeed);
      } else if (action === "duck") {
        if (trex.jumping) {
          if (!trex.speedDrop) trex.setSpeedDrop();
        } else if (!trex.ducking) {
          trex.setDuck(true);
        }
      } else {
        if (trex.jumping) trex.endJump();
        // Both keys are released in the idle action; native duck keyup
        // clears speedDrop even while the dino is still in the air.
        if (trex.speedDrop) trex.speedDrop = false;
        if (trex.ducking) trex.setDuck(false);
      }

      this.trackPassed(wasDucking);
    },

    trackPassed(wasDucking) {
      const trex = runner.tRex;
      for (const obstacle of runner.horizon.obstacles) {
        if (!obstacle.__passed && obstacle.xPos + obstacle.width < trex.xPos) {
          obstacle.__passed = true;
          this.passed++;
          if (obstacle.typeConfig.type === "PTERODACTYL") {
            this.pterodactyls_passed++;
            const duckable = obstacle.yPos >= 60 && obstacle.yPos < 90;
            if (duckable) {
              this.pterodactyls_duckable++;
              // Use the pose from before this frame's action: that is the
              // pose the game used while the pterodactyl moved past.
              if (wasDucking) this.pterodactyls_ducked++;
            }
          }
        }
      }
    },

    finish() {
      this.active = false;
      this.done = true;
    },

    start(net) {
      this.net = net;
      this.active = false;
      this.done = false;
      this.started = performance.now();
      this.frames = 0;
      this.passed = 0;
      this.last_action = "idle";
      this.action_counts = { jump: 0, duck: 0, idle: 0 };
      this.useful_duck_frames = 0;
      this.useless_duck_frames = 0;
      this.bad_jump_frames = 0;
      this.useless_jump_frames = 0;
      this.pterodactyls_passed = 0;
      this.pterodactyls_duckable = 0;
      this.pterodactyls_ducked = 0;

      // Start a clean real-time game. No clock freezing or frame stepping.
      runner.stop();
      runner.activated = true;
      runner.playingIntro = false;
      runner.paused = false;
      runner.crashed = false;
      runner.tRex.playingIntro = false;
      runner.tRex.xPos = runner.tRex.config.START_X_POS;
      this.active = true;
      runner.restart();

      // Normal play reaches this through playIntro/startGame. We skip the
      // intro so every genome starts instantly. Do the same container setup
      // and arcade-mode scaling the intro would have done, otherwise the
      // 44px-wide waiting container clips the 600px canvas to a thin sliver.
      // This is presentation only: game speed, physics and the gradual
      // difficulty ramp are untouched.
      runner.containerEl.style.width = runner.dimensions.WIDTH + "px";
      runner.containerEl.style.height = runner.dimensions.HEIGHT + "px";
      if (runner.setArcadeMode) runner.setArcadeMode();
      const messageBox = document.getElementById("messageBox");
      if (messageBox) messageBox.style.display = "none";
    },

    stop() {
      this.active = false;
      this.done = true;
      runner.stop();
    },
  };

  // One wrapper per page: run the game at its intended 60 FPS and let the
  // network act after every real game frame. Headless Chromium can fire rAF
  // much faster than 60 Hz; without this cap the game's per-frame difficulty
  // ramp would accelerate. This is not a training speed hack and the original
  // update() still does all game logic. The 1 ms look-ahead absorbs normal
  // 60 Hz jitter without letting the long-term rate exceed 60 updates/s.
  const FRAME_MS = 1000 / 60;
  let nextFrame = 0;
  const originalUpdate = runner.update.bind(runner);
  runner.update = function () {
    const now = performance.now();
    if (!nextFrame) nextFrame = now;
    if (now + 1.0 < nextFrame) {
      runner.updatePending = false;
      runner.scheduleNextUpdate();
      return;
    }
    nextFrame += FRAME_MS;
    if (nextFrame < now) nextFrame = now + FRAME_MS;
    originalUpdate();
    if (agent.active) agent.step();
  };

  runner.playSound = function () {};
  window.__dinoAgent = agent;
})();
