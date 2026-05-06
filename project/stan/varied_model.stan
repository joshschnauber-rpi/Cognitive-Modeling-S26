data {
    // Sizes
    int<lower=1> N; // The total number of trials
    // Experiment
    array[N] int<lower=1, upper=12> main_input_order; // The order being guessed (spatial, temporal)
    array[N] int<lower=1, upper=12> other_input_order; // The order not being guessed (temporal, spatial)
    // Results
    array[N] int<lower=1, upper=12> recall_order; // The recall for the main_input_order
}


parameters {
    array[12] real<lower=0, upper=1> p_main_correct;
    real<lower=1e-6> sigma_main_recall;
    array[12] real<lower=1e-6> sigma_other_recall;
}


model {
    // Priors
    p_main_correct ~ beta(2, 2);
    sigma_main_recall ~ lognormal(0, 0.5);
    sigma_other_recall ~ lognormal(-1, 0.5);

    // Likelihood
    for( i in 1:N ) {
        int main_pos = main_input_order[i];
        int other_pos = other_input_order[i];
        int recall_pos = recall_order[i];
        
        // If the guess was correct
        // Certain probability of guessing correctly
        if( recall_pos == main_pos ) {
            target += log(p_main_correct[main_pos]);
        }
        // If the guess was not correct
        // Guess randomly using a normal dist around the correct pos
        else {
            real sigma_new = exp(sigma_main_recall + sigma_other_recall[other_pos]);
            real log_normal_density = log_diff_exp(
                normal_lcdf(recall_pos + 0.5 | main_pos, sigma_new),
                normal_lcdf(recall_pos - 0.5 | main_pos, sigma_new)
            );
            target += log((1 - p_main_correct[main_pos])) + log_normal_density;
        }
    }
}


generated quantities {
    array[N] int<lower=1, upper=12> recall_order_gen;

    // Precompute pdfs at all positions
    array[12, 12] simplex[12] log_pdfs;
    for( o_pos in 1:12 ) {
        real sigma_new = exp(sigma_main_recall + sigma_other_recall[o_pos]);
        for( m_pos in 1:12 ) {
            vector[12] log_pdf;
            for( p in 1:12 ) {
                if( p == m_pos ) {
                    log_pdf[p] = negative_infinity();
                }
                else {
                    real log_normal_density = log_diff_exp(
                        normal_lcdf(p + 0.5 | m_pos, sigma_new),
                        normal_lcdf(p - 0.5 | m_pos, sigma_new)
                    );
                    log_pdf[p] = log_normal_density;
                }
            }
            log_pdfs[o_pos, m_pos] = softmax(log_pdf);
        }
    }

    for( i in 1:N ) {
        int main_pos = main_input_order[i];
        int other_pos = other_input_order[i];

        if( bernoulli_rng(p_main_correct[main_pos]) ) {
            recall_order_gen[i] = main_pos;
        } 
        else {
            recall_order_gen[i] = categorical_rng(log_pdfs[other_pos, main_pos]);
        }
    }
}