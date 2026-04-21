# Game NCA

Neural cellular automata trained to control small differentiable games. This is inspired by the classic NCA setup where each cell reads its local neighborhood, updates hidden signaling channels, and learns useful global behavior through many local updates.

## Idea

Here, the automaton also gets read-only game-pixel channels. At every game tick:

1. The game renders pixels into observation channels.
2. The NCA runs a few asynchronous local updates over internal state plus game pixels.
3. One internal channel is treated as an action field.
4. A smoothed max / soft-argmax reads the brightest position from that channel.
5. The game advances using that coordinate as the paddle target.

Training uses a state pool, so sampled automata/game pairs are at different evolution ages.

## Games

- `pong`: one-player differentiable Pong. The NCA reads ball, paddle, and wall pixels. The right-side brightness of channel 1 chooses the target `y` for the paddle.
- `catch`: a smaller curriculum game. A falling target must be caught by a bottom paddle. The bottom brightness of channel 1 chooses target `x`.
- `maze`: random perfect mazes. The NCA grows a one-channel value field from the goal through corridors, and an agent greedily follows the brightest neighboring cell.

The loss combines game-derived tracking pressure with a distributional loss on the action channel. There are no CLIP or style losses.

## Run

Use the existing Python environment:

```bash
python3 -m nca_game.train --config configs/catch.yaml --out runs/catch
python3 -m nca_game.train --config configs/pong.yaml --out runs/pong
python3 -m nca_game.train_maze --config configs/maze.yaml --out runs/maze
```

On a Mac with MPS available, `--device auto` will choose it. You can force CPU with:

```bash
python3 -m nca_game.train --config configs/pong.yaml --out runs/pong_cpu --device cpu
```

Render a checkpoint:

```bash
python3 -m nca_game.render runs/pong/checkpoints/latest.pt
```

Evaluate a checkpoint:

```bash
python3 -m nca_game.evaluate runs/pong/checkpoints/latest.pt --episodes 64 --steps 300
python3 -m nca_game.evaluate_maze runs/maze/checkpoints/latest.pt --size 31 --count 128
```

## Outputs

Each run writes:

- `config.resolved.yaml`
- `train_log.jsonl`
- `checkpoints/latest.pt`
- `videos/final_<game>_rollout.mp4` or `.gif`

The rollout video shows the game pixels, the action channel with predicted and target guide lines, the action distribution, and a grid of internal NCA signaling channels.

## Notes

This is intentionally compact and hackable. Good next experiments:

- Remove the action-channel cross-entropy and rely more heavily on sparse hit/miss pressure.
- Add a two-paddle adversary where a frozen or heuristic left paddle returns the ball.
- Increase `state_channels` and reduce `nca_steps_per_tick` to see whether the automaton learns a faster communication protocol.
- Save pool snapshots and compare young versus old automata states.
