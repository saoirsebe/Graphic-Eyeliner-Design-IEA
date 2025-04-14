from matplotlib import pyplot as plt


def test_performance(monkeypatch, benchmark):
    import InitialiseGenePool
    monkeypatch.setattr(InitialiseGenePool, "initial_gene_pool_size", 100)
    # Benchmark the function
    stats = benchmark.pedantic(InitialiseGenePool.initialise_gene_pool, rounds=50, iterations=1)

    # Extract timing data
    times = benchmark.stats['data']
    average_time = sum(times) / len(times)
    print(f"\nAverage time over {len(times)} runs: {average_time:.6f} seconds")

    # Box plot (not saved)
    plt.boxplot(times, vert=False, patch_artist=True)
    plt.title("Function initial_gene_pool() Time Taken Box Plot")
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


