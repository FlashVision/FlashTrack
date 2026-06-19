"""Example: Run multi-object tracking on a video file."""

from flashtrack import Predictor


def main():
    predictor = Predictor(
        model_path="workspace/model_best_inference.pth",
        tracker_type="bytetrack",
        track_thresh=0.5,
        track_buffer=30,
    )

    output_path = predictor.track_video(
        video_path="input.mp4",
        output_dir="output/",
        show=False,
    )
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
