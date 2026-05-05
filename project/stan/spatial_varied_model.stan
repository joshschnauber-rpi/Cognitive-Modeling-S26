data {
    // Sizes
    int<lower=1> N;
    int<lower=1> S;
    // IDs
    array[N] int<lower=1, upper=S> subject_index;
    // Experiment
    array[N] real<lower=1> spatial_input_order;
    // Results
    array[N] real<lower=1> spatial_recall_order;
    //array[N] int<lower=1> output_rank;
    array[N] real<lower=0> rt;
}


parameters {
    real<lower=1e-6> sigma_recall;

    real mu_rt;
    real<lower=1e-6> sigma_rt;
    real gamma_rt;
}


model {
    // Priors
    sigma_recall ~ lognormal(1, 0.5);

    mu_rt ~ normal(0.5, 0.5);
    sigma_rt ~ lognormal(-1, 0.5);
    gamma_rt ~ normal(1, 1);

    // Likelihood
    for( i in 1:N ) {
        // The recall is expected to within some distance of actual
        spatial_recall_order[i] ~ normal(spatial_input_order[i], sigma_recall);
        real dist = abs(spatial_recall_order[i] - spatial_input_order[i]);

        // The lower the dist, the faster the recall
        rt[i] ~ lognormal(mu_rt + gamma_rt * dist, sigma_rt);
    }
}