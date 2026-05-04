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
    array[S] real<lower=0> subject_sigma;
    vector[N] recall_latent;

    real alpha;
    real<lower=0> beta;

    real mu_rt;
    real<lower=0> sigma_rt;
    real gamma;
}


model {
    for( i in 1:S ) {
        subject_sigma[i] ~ normal(1, 1);
    }

    for( i in 1:N ) {
        int s_i = subject_index[i];

        recall_latent[i] ~ normal(spatial_input_order[i], subject_sigma[s_i] * list_length[i]);

        real dist = abs(recall_latent[i] - spatial_input_order[i]);

        correct[i] ~ bernoulli_logit(alpha - beta * dist);

        rt[i] ~ lognormal(mu_rt + gamma * dist, sigma_rt);
    }
}


// parameters {
//     array[S] real<lower=0> subject_sigma;
// }

// model {
//     for( i in 1:S ) {
//         subject_sigma[i] ~ normal(0.2, 0.1);
//     }

//     for( i in 1:N ) {
//         // Get index of subject for this word
//         int s_i = subject_index[i];

//         //
        
//         // The recall is expected to within some distance of actual
//         spatial_recall_order[i] ~ normal(spatial_input_order[i], subject_sigma[s_i] * list_length[i]);
//         // If the recall and input are equal, the guess is correct
//         distance_from_correct[i] ~ normal(abs(spatial_input_order[i] - spatial_recall_order[i]), 0);
//         if( distance_from_correct[i] == 0 ) {
//             correct[i] ~ normal(1, 0);
//         }
//         else {
//             correct[i] ~ normal(0, 0);
//         }

//         // T
//     }
// }