"""Example: Benchmark FlashTrack model speed and size."""

from flashtrack.analytics import Benchmark, Profiler


def main():
    # Benchmark latency
    for size in ["m-0.5x", "m", "m-1.5x"]:
        bench = Benchmark(model_size=size, input_size=(128, 64), device="cpu")
        results = bench.run(warmup=10, iterations=50)
        print(f"\n{size}:")
        print(f"  Latency: {results['latency_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.1f}")
        print(f"  Params: {results['params_m']:.2f}M")
        print(f"  FP16 size: {results['model_fp16_mb']:.2f} MB")

    # Detailed profiling
    print("\n" + "=" * 70)
    profiler = Profiler(model_size="m")
    profiler.print_report()


if __name__ == "__main__":
    main()
