"""
Example script to load a saved population and plot observability helpers.

Update `population_path` before running:
    python -m meta_train.population_based.scripts.plot_population_observability
"""
from pathlib import Path

from meta_train.population_based.observability import (
    plot_scores_over_steps,
    plot_survival_by_mutation_operator,
    compute_score_standard_error_stats,
    plot_population_pca_embedding,
)
from meta_train.population_based.population import Population


if __name__ == "__main__":
    # TODO: update this path to your saved population pickle
    population_path = Path("logs/first_run_cat_and_dog_w_env_feedback_continued/20251230_195358/population.pkl")
    population = Population.load(population_path)

    mean_standard_error, standard_error_of_mean = compute_score_standard_error_stats(population)
    print(f"Mean standard error: {mean_standard_error:.2f} ± {standard_error_of_mean:.2f}")

    fig_scores = plot_scores_over_steps(population)
    fig_survival_ignore_alive = plot_survival_by_mutation_operator(
        population, treat_alive_as_dead=False
    )
    fig_survival_alive_as_dead = plot_survival_by_mutation_operator(
        population, treat_alive_as_dead=True
    )

    # Customize/save as needed; here we just show them
    # fig_scores.show()
    # fig_survival_ignore_alive.show()
    # fig_survival_alive_as_dead.show()

    # Or, to save:
    fig_scores.savefig("scores_over_steps.png", dpi=300, bbox_inches="tight")
    fig_survival_ignore_alive.savefig("survival_ignore_alive.png", dpi=300, bbox_inches="tight")
    fig_survival_alive_as_dead.savefig("survival_alive_as_dead.png", dpi=300, bbox_inches="tight")

    # Interactive PCA embedding visualization (saved as HTML)
    fig_pca = plot_population_pca_embedding(population)
    fig_pca.write_html("population_pca_embedding.html")

