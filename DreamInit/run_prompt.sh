#!/bin/bash

# 定義你要跑的 prompt 清單
PROMPTS=(
  "A rabbit is cutting grass with a lawnmower."
  "A humanoid banana sitting at a desk doing homework."
  "A squirrel knight in armor jousting on a lawn."
  "A clockwork engineer repairing the gears of a massive steam-powered machine."
  "A couple cooking a complex dinner together."
  "A koala wearing a party hat and blowing out candles."
  "A robot single-handedly lifting a basketball."
  "A squirrel gesturing in front of an easel showing colorful pie charts."
  "A stylish fox typing on a vintage typewriter."
  "A whale breaching the ocean surface and splashing back down."
  "An individual sitting on a park bench, scrolling through his smartphone."
  "Two bears sharing a jar of honey while sitting on a log."
  "Two llamas wearing bow ties and playing chess."
  "Two owls playing tic-tac-toe with sticks and stones."
)

# 迴圈跑每個 prompt
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  WORKSPACE="workspace/test_$i"

  echo "Running with prompt: $PROMPT"

  CUDA_VISIBLE_DEVICES=0 python main.py \
    --prompt "$PROMPT" \
    --workspace "$WORKSPACE" \
    --port $((7348 + i)) \
    --fp16 \
    --perpneg \
    --lr 7e-5 \
    --test_interval 5
done
