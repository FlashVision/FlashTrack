"""Example: Export ReID model to ONNX format."""

from flashtrack import Exporter


def main():
    exporter = Exporter(model_path="workspace/model_best_inference.pth")
    output_path = exporter.export(
        output="reid_model.onnx",
        simplify=True,
    )
    print(f"ONNX model exported to: {output_path}")


if __name__ == "__main__":
    main()
