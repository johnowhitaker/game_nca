# Experiment Log

## 2026-04-21

All runs used PyTorch on Apple MPS.

### Pong smoke

Command:

```bash
python3 -m nca_game.train --config configs/pong.yaml --out runs/smoke_pong --steps 3 --device cpu --no-render
```

Evaluation:

```bash
python3 -m nca_game.evaluate runs/smoke_pong/checkpoints/latest.pt --device mps --episodes 128 --steps 320 --burn-in 30
```

Result:

- Mean coordinate error: `0.4008`
- Hits: `214`
- Misses: `501`
- Hit rate: `0.2993`

### Pong, first MPS pass

Command:

```bash
python3 -m nca_game.train --config configs/pong.yaml --out runs/pong_mps_800 --steps 800 --device mps
```

Evaluation:

```bash
python3 -m nca_game.evaluate runs/pong_mps_800/checkpoints/latest.pt --device mps --episodes 128 --steps 320 --burn-in 30
```

Result:

- Mean coordinate error: `0.3364`
- Hits: `342`
- Misses: `222`
- Hit rate: `0.6064`

### Pong, reset-aware baseline

During this run, the trainer reset the NCA hidden state whenever a game episode reset after a miss. This made the game-state and automaton-state lifecycles match.

Command:

```bash
python3 -m nca_game.train --config configs/pong.yaml --out runs/pong_reset_1000 --steps 1000 --device mps
```

Evaluation:

```bash
python3 -m nca_game.evaluate runs/pong_reset_1000/checkpoints/latest.pt --device mps --episodes 128 --steps 320 --burn-in 30
```

Result:

- Mean coordinate error: `0.3383`
- Hits: `415`
- Misses: `70`
- Hit rate: `0.8557`

Artifact:

- `runs/pong_reset_1000/videos/final_pong_rollout.gif`
- `runs/pong_reset_1000/videos/final_pong_contact_sheet.png`

### Catch curriculum

Command:

```bash
python3 -m nca_game.train --config configs/catch.yaml --out runs/catch_short --steps 120 --device mps
```

This run was useful as a smoke test for the x-axis action readout, but 120 steps did not produce stable learning. The Catch task likely needs either a longer curriculum or a stronger early action-channel loss.

## Notes

- The reset-aware Pong checkpoint is the best current artifact.
- The action-channel cross-entropy does decrease slowly; hit rate improved more clearly than coordinate error, which suggests the paddle-control dynamics are forgiving once the action field is roughly in the right vertical neighborhood.
- A useful next run would anneal the action-readout temperature and train with a persistent pool that resets NCA state only on game resets.

## Maze NCA

The maze experiment changes the readout from a global paddle coordinate to a local slime-mold-style value field. The NCA observes three channels: walls, goal, and start. One internal channel is trained as a pheromone/value field on the optimal start-to-goal path. During rollout, the agent samples neighboring cells from that value field and prefers unvisited cells, which prevents trivial bouncing while still letting the learned field guide exploration.

### First attempt

The first curriculum used full BFS distance targets on every open cell with stochastic CA updates and too few NCA propagation steps. It produced smooth fields but no greedy solves:

- `15x15`: about `0%` success in live eval after the short stage
- `21x21`: about `0%`
- `31x31`: about `0%`

The core issue was propagation budget. A 31x31 perfect maze can have an optimal path around 240 cells long; a local automaton given 50-70 updates cannot propagate reliable goal information that far.

### Path-focused target

Changes:

- Train only the optimal path as the high-value field, with non-path open cells suppressed.
- Mask direction loss to the optimal path.
- Use deterministic updates for mazes: `update_rate=1.0`.
- Increase NCA updates with maze size.
- Use a visited-cell trail during rollout, so the agent explores without two-cell loops.

Training commands:

```bash
python3 -m nca_game.train_maze --config configs/maze.yaml --out runs/maze15_long --steps 2000 --device mps
python3 -m nca_game.train_maze --config configs/maze_21.yaml --out runs/maze21_finetune --steps 1600 --device mps --resume runs/maze15_long/checkpoints/latest.pt
```

Evaluation of `runs/maze21_finetune/checkpoints/latest.pt`:

```bash
python3 -m nca_game.evaluate_maze runs/maze21_finetune/checkpoints/latest.pt --device mps --size 15 --count 256 --nca-steps 86
python3 -m nca_game.evaluate_maze runs/maze21_finetune/checkpoints/latest.pt --device mps --size 21 --count 256 --nca-steps 145
python3 -m nca_game.evaluate_maze runs/maze21_finetune/checkpoints/latest.pt --device mps --size 31 --count 128 --nca-steps 260
```

Results:

- `15x15`: success rate `0.4648`, mean path ratio `2.2786`
- `21x21`: success rate `0.4141`, mean path ratio `2.2939`
- `31x31`: success rate `0.2734`, mean path ratio `2.5274`

Video:

```bash
python3 -m nca_game.render_maze runs/maze21_finetune/checkpoints/latest.pt --out runs/maze21_finetune/videos/maze31_best_of_16.mp4 --device mps --size 31 --candidates 16 --nca-steps 260 --extra-steps 120 --capture-every 3 --move-every 2 --fps 24
```

The renderer sampled 16 random 31x31 mazes and chose the longest solved candidate first:

- Best seed: `1151485119`
- Size: `31x31`
- Optimal path length: `294`
- Rendered path length: `294`
- Artifact: `runs/maze21_finetune/videos/maze31_best_of_16.mp4`

## Glow Garden Flow NCA

This was a more playful follow-up to the maze result. Instead of a single maze agent, random cave-like maps contain blue source cells and green flower cells. The NCA observes walls, sources, and flowers, then grows one internal channel into a flower potential field. Many particles spawn from the sources and follow the local gradient of that learned field, leaving orange/yellow trails.

Training command:

```bash
python3 -m nca_game.train_flow --config configs/flow.yaml --out runs/flow_garden --device mps
```

Training was much easier than the brittle maze task because the cave fields are open and many local paths are acceptable. The direction accuracy reached roughly `95-97%` in the final third of training.

Mid-run render:

```bash
python3 -m nca_game.render_flow runs/flow_garden/checkpoints/step_00500.pt --out runs/flow_garden/videos/glow_garden_500.mp4 --device mps --size 45 --seed 2027 --steps 360 --warmup 125 --particles 110 --capture-every 2 --fps 24
```

Final best-of render:

```bash
python3 -m nca_game.render_flow runs/flow_garden/checkpoints/latest.pt --out runs/flow_garden/videos/glow_garden_best_of_8.mp4 --device mps --size 45 --seed 4242 --candidates 8 --steps 480 --warmup 130 --particles 130 --capture-every 2 --fps 24
```

Result:

- Best seed: `1324107816`
- Candidate pre-score visits: `513`
- Rendered flower visits: `566`
- Artifact: `runs/flow_garden/videos/glow_garden_best_of_8.mp4`

This is currently the most visually pleasant “toy ecosystem” artifact: particles stream along the learned luminous field while the hidden channels show broad propagating bands and obstacle shadows.
