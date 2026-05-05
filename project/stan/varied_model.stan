data {
    // Sizes
    int<lower=1> N; // The total number of trials
    // Experiment
    array[N] int<lower=1, upper=12> main_input_order; // The order being guessed (spatial, temporal)
    array[N] int<lower=1, upper=12> other_input_order; // The order not being guessed (temporal, spatial)
    // Results
    array[N] real<lower=1, upper=12> recall_order; // The recall for the main_input_order
}


parameters {
    real<lower=1e-6> sigma_base_recall;
    array[12] real<lower=(-sigma_base_recall+1e-6)> beta_other_order;
}


model {
    // Priors
    sigma_base_recall ~ lognormal(0.5, 0.5);
    for( i in 1:12 ) {
        beta_other_order[i] ~ normal(0, 1);
    }

    // Likelihood
    for( i in 1:N ) {
        // The recall is expected to within some distance of input order to recall
        // But the accuracy may also depend on the position in the other ordering
        real sigma_new = sigma_base_recall + beta_other_order[other_input_order[i]];
        recall_order[i] ~ normal(main_input_order[i], sigma_new);
    }
}


generated quantities {
    array[N] real<lower=1, upper=12> recall_order_gen;

    for (i in 1:N) {
        real sigma_new = sigma_base_recall + beta_other_order[other_input_order[i]];
        // Sample until in range
        real recall_order_f = normal_rng(main_input_order[i], sigma_new);
        int recall_order_int = to_int(round(recall_order_f));
        recall_order_gen[i] = fmin(fmax(recall_order_int, 1), 12);
    }
}