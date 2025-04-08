from matplotlib import pyplot as plt


def test_performance(monkeypatch, benchmark):
    import InitialiseGenePool
    monkeypatch.setattr(InitialiseGenePool, "initial_gene_pool_size", 200)
    # Benchmark the function
    result = benchmark(InitialiseGenePool.initialise_gene_pool)
    times = benchmark.stats['data']
    average_time = sum(times) / len(times)
    print(f"\nAverage time over {len(times)} runs: {average_time:.6f} seconds")

    # Plot histogram (not saved)
    plt.hist(times, bins=20)
    plt.title("InitialiseGenePool Timing Histogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Optional: assert something about the result
    assert result is not None

