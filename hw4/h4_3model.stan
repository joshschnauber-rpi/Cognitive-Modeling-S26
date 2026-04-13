data {
    int<lower=0> N;
    int<lower=0> K;
    int<lower=0> KM;
    matrix[N, K] xm; //rankings
    array[N] int<lower=0, upper=1> ym;    //yes no binary
    matrix[KM, K] xtest;
    array[KM] int<lower=0, upper=1> ytest;
}

parameters {
    real alpha;     //offset
    vector[K] beta; //linear modifications
    //real<lower=0> sigma; //possible error
}

//transformed parameters { }

model {

    alpha ~ normal(0,2); //since y is between 0-1 the offset is probably very small
    beta ~ normal(0,2);  //same for the beta modifications, the prior would be small
    //sigma ~ normal(0,1) 
    target += bernoulli_logit_lpmf(ym | alpha + xm*beta);
    //ym ~ normal(rows_dot_product(beta, xm), sigma);
    
    
}

generated quantities {
    array[KM] int y_sim;
    for (n in 1:KM){
        y_sim[n] = bernoulli_logit_rng(xtest[n]*beta + alpha);
    }

    real<lower=0, upper=1> mean = sum(y_sim) * 1.0 / KM;
    
    real temp = 0;
    real temp2 = 0;
    real temp3 = 0;
    for (n in 1:KM){
        temp += y_sim[n]-ytest[n];
        temp2 += 0.5 - ytest[n];
        temp3 += mean - ytest[n];
    }

    real test_against = temp / KM;
    real class_acc = temp2 / KM;
    real single_mean_test = temp3 / KM;

}
