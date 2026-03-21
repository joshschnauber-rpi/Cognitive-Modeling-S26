data {
    int<lower=1> N; // number of rows
    int<lower=1> J; // number of people
    array[N] int<lower=1, upper=J> id;   // person ID for each trial/row
    array[N] real<lower=0> y; // response time
    array[N] int<lower=1, upper=2> condition; // condition indicator
    array[N] int<lower=0, upper=1> choice; // choice indicator
}

transformed data {
  // init a vector to hold the minimum response times
  real eps = 1e-6;
  array[J] real rt_min;

  // initialize to +inf
  for (j in 1:J) {
    rt_min[j] = positive_infinity();
  }

  // compute per-person minimum RT
  for (n in 1:N) {
    if (y[n] < rt_min[id[n]]) {
        rt_min[id[n]] = y[n];
    }
  }
}

parameters {
    // Each person has their own parameters
    array[J] real v_cond1;
    array[J] real v_cond2;
    array[J] real<lower=0> a;
    array[J] real<lower=0, upper=1> beta;
    array[J] real<lower=0, upper=1> tau_raw;
}

transformed parameters {
  // A good way to bound non-decision time (tau)
  array[J] real<lower=0> tau;
  for (j in 1:J) {
    tau[j] = tau_raw[j] * (rt_min[j] - eps);
  }
}

model {
    // Priors
    for (j in 1:J) {
        v_cond1[j] ~ normal(0, 1);
        v_cond2[j] ~ normal(0, 1);
        a[j] ~ lognormal(0, 1);
        beta[j] ~ beta(5, 5);
        tau_raw[j] ~ lognormal(0, 1);
    }

    // Likelihood
    for (n in 1:N) {
        // Condition 1
        if (condition[n] == 1) {
            if (choice[n] == 1) {
                y[n] ~ wiener(a[id[n]], tau[id[n]], beta[id[n]], v_cond1[id[n]]);
            } else {
                y[n] ~ wiener(a[id[n]], tau[id[n]], 1-beta[id[n]], -v_cond1[id[n]]);
            }
        }

        // Condition 2
        if (condition[n] == 2) {
            if (choice[n] == 1) {
                y[n] ~ wiener(a[id[n]], tau[id[n]], beta[id[n]], v_cond2[id[n]]);
            } else {
                y[n] ~ wiener(a[id[n]], tau[id[n]], 1-beta[id[n]], -v_cond2[id[n]]);
            }
        }
    }
}