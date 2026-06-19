# FAQ

## General

**Q: What datasets does FlashTrack support?**
A: Any dataset in MOT17/MOT20 format (images + gt.txt annotations). Convert your data to this format for custom tracking.

**Q: What's the difference between FlashTrack and FlashDet?**
A: FlashDet is for object detection (finding objects in frames). FlashTrack is for multi-object tracking (maintaining identity across frames). They share the same ShuffleNetV2 backbone.

**Q: Can I use FlashTrack without a ReID model?**
A: Yes! ByteTracker and SORTTracker work without ReID features. Only DeepSORTTracker benefits from ReID.

## Training

**Q: How much data do I need?**
A: The MOT17 training split (~5K identities) works well. For custom domains, at least 100+ identities with 10+ samples each.

**Q: Which LoRA variant should I use?**
A: Start with `standard`. Use `dora` for higher quality, `lora_fa` for minimal memory, or `lora_plus` for faster convergence.

**Q: How do I resume training?**
A: Use `Trainer(resume="workspace/checkpoint_last.pth")` or `--resume` flag.

## Deployment

**Q: How do I export to ONNX?**
A: `flashtrack export --model best.pth --output reid.onnx --simplify`

**Q: What's the smallest model?**
A: FlashTrack-m-0.5x at ~0.3M params / ~0.6 MB FP16.

**Q: Can I run on CPU?**
A: Yes, use `--device cpu`. CPU inference is fast enough for real-time with ByteTracker.

## Troubleshooting

**Q: `scipy` import error?**
A: Install scipy: `pip install scipy>=1.10.0`

**Q: CUDA out of memory?**
A: Reduce batch size, enable `amp=True`, or use LoRA/QLoRA.

**Q: Track IDs keep changing?**
A: Increase `track_buffer`, lower `match_thresh`, or switch to DeepSORTTracker with ReID features.
