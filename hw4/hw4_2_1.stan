data {
    int<lower=0> N;                             // Number of words
    array[N] int<lower=0, upper=1> old;         // Whether the word is in the old list or not (0 = new, 1 = old, )
    array[N] int<lower=0, upper=1> response;   // Whether the participant though the word was in the old list or not (0 = new, 1 = old)
}

parameters {
    real<lower=0, upper=1> d;
    real<lower=0, upper=1> g;
}

model {
    // Priors
    d ~ beta(5, 5);
    g ~ beta(5, 5);

    // Likelihood
    for( i in 1:N ) {
        // Word in old list
        if( old[i] == 1) {
            response[i] ~ bernoulli(d + (1-d) * g);
        }
        // Word in new list
        else if( old[i] == 0) {
            response[i] ~ bernoulli(g);
        }
    }
}