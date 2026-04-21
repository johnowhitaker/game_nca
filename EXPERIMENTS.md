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
