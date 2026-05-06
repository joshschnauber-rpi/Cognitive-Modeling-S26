data {
    // Sizes
    int<lower=1> N;
    // Experiment
    array[N] real<lower=0, upper=1> spatial_input_order;
    array[N] real<lower=-1, upper=1> temporal_input_pos;
    array[N] int<lower=0> lengthh;
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
    //like it said in the paper experiment 3 is a 2x2x3 study
    //where some are spatial/temporal and aloud/silent and fixed item information/varied item information
    //thus each combination gets us 8
    //then the params would be alpha + b1*(spatial input order-6.5)^2 + b2*(Temporal_Input_Position) 
    //which would relate to correct[i] for the corresponding type of the 2x2x2 that the value resides.
    //then those 3(a,b1,b2) are repeated for relating to response time except b1 is now relating to a linear instead of a shifted square
    array[4,3,3] real params;
}

model {

    for( i in 1:4 ){
        for(j in 1:3){
        target += normal_lpdf(params[i,j,1] | 0, 2);  //alpha
        target += normal_lpdf(params[i,j,2] | 0, 2);  //b1 (spatial)
        target += normal_lpdf(params[i,j,3] | 0, 2);  //b2 (temporal)
        }
    }

    for( i in 1:N ){
        //real a1 = params[type[i]][length[i]][1];
        //real b1 = params[type[i]][length[i]][2];
        //real b2 = params[type[i]][length[i]][3];

    
        //target += bernoulli_logit_lupmf(correct[i] | (params[type[i] + (length[i]-4)/4][1] + params[type[i] + (length[i]-4)/4][2]*pow((spatial_input_order[i]-6.5)/6.5,2) + params[type[i] + (length[i]-4)/4][3]*(temporal_input_pos[i]-6.5)/6.5));

        target += bernoulli_logit_lpmf(correct[i] | (params[type[i],lengthh[i],1] + params[type[i],lengthh[i],2]*spatial_input_order[i] + params[type[i],lengthh[i],3]*temporal_input_pos[i]));
        
    }


}

