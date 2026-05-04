data {
    // Sizes
    int<lower=1> N;
    int<lower=1> S;
    // IDs
    array[N] int subject_index;
    // Experiment
    //array[N] int<lower=1> list_number;
    array[N] int<lower=1> spatial_input_order;
    //array[N] int<lower=1> temporal_input_pos;
    array[N] int<lower=1> list_length;
    // Results
    array[N] int<lower=1> spatial_recall_order;
    array[N] int<lower=0> distance_from_correct;
    array[N] int<lower=0, upper=1> correct;
    //array[N] int<lower=1> output_rank;
    array[N] real<lower=0> rt;
}

transformed data {
   //array[N] int<lower=1, upper=N> rank_sorted_indexes = sort_indices_asc(output_rank);
}


parameters {
    array[S] real<lower=0> sigma_recall;
    array[N] real recall;

    real correct_alpha;
    real<lower=0> correct_beta;

    real mu_rt;
    real<lower=0> sigma_rt;
    real gamma;
}


model {
    for( i in 1:S ) {
        sigma_recall[i] ~ normal(1, 1);
    }

    for( i in 1:N ) {
        int s_i = subject_index[i];

        // The recall is expected to within some distance of actual
        // Anc recall gets less accurate as the size of the list increases
        recall[i] ~ normal(spatial_input_order[i], sigma_recall[s_i] * log(list_length[i]));
        real dist = abs(recall[i] - spatial_input_order[i]);

        // The lower the dist, the more likely it is to be correct
        correct[i] ~ bernoulli_logit(correct_alpha - correct_beta * dist);

        rt[i] ~ lognormal(mu_rt + gamma * dist, sigma_rt);
    }
}