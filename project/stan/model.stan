data {
    // Sizes
    int<lower=1> N;
    // Experiment
    array[N] int<lower=1> spatial_input_order;
    array[N] int<lower=1> temporal_input_pos;
    // Results
    array[N] int<lower=0, upper=1> correct;
    array[N] real<lower=0> rt;
    array[N] int type;
}

transformed data {
   //array[N] int<lower=1, upper=N> rank_sorted_indexes = sort_indices_asc(output_rank);
}


parameters {
    //This is a list of priors that are for each type of study
    //like it said in the paper experiment 3 is a 2x2x2 study
    //where some are spatial/temporal and aloud/silent and fixed item information/varied item information
    //thus each combination gets us 8
    //then the params would be alpha + b1*(spatial input order-6.5)^2 + b2*(Temporal_Input_Position) 
    //which would relate to correct[i] for the corresponding type of the 2x2x2 that the value resides.
    //then those 3(a,b1,b2) are repeated for relating to response time except b1 is now relating to a linear instead of a shifted square
    array[8,3] real params;
    real<lower=0> sigma;
}


model {
    //sigma ~ normal(10,5);
    for( i in 1:8 ){
        target += normal_lupdf(params[i][1] | 1, 1);  //alpha
        target += normal_lupdf(params[i][2] | 1, 1);  //b1 (spatial)
        target += normal_lupdf(params[i][3] | 1, 1);  //b2 (temporal)
        //target += normal_lupdf(params[i][4] | 10, 10); //alpha for time
        //target += normal_lupdf(params[i][5] | 3, 5); //b1 for time
        //target += normal_lupdf(params[i][6] | 3, 5);  //b2 for time
    }

    for( i in 1:N ){
        //real a1 = params[type[i]][1];
        //real b1 = params[type[i]][2];
        //real b2 = params[type[i]][3];
        //real at2 = params[type[i]][4];
        //real bt1 = params[type[i]][5];
        //real bt2 = params[type[i]][6];
    
        target += bernoulli_logit_lupmf(correct[i] | (params[type[i]][1] + params[type[i]][2]*pow((spatial_input_order[i]-6.5)/6.5,2) + params[type[i]][3]*(temporal_input_pos[i]-6.5)/6.5));
        //target += normal_lupdf(rt[i] | (params[type[i]][4] + params[type[i]][5]*spatial_input_order[i] + params[type[i]][6]*temporal_input_pos[i]), sigma);
    }


}

