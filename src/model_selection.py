def select_best_model(results):

    best_name = None
    best_model = None
    best_r2 = float("-inf")

    for name, result in results.items():

        metrics = result["metrics"]

        if metrics["R2"] > best_r2:

            best_r2 = metrics["R2"]
            best_name = name
            best_model = result["model"]

    return {
        "name": best_name,
        "model": best_model,
        "r2": best_r2
    }
