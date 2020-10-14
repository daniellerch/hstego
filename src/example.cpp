#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "stc_ml_c.h"

int main() {

    const float payload = 0.5; // 
    const uint n = 1e+5;       // cover size
    uint h = 10;               // constraint height of STC code
    uint m = floor(payload*n); // number of message bits to embed
    uint trials = 10;          // if the message cannot be embedded due to large amount of 
                               // wet pixels, then try again with smaller message. Try at most 10 times.
    
    srand(0);
    // srand(time(NULL));

    // generate random 8bit cover
    int* cover = (int*)malloc(n*sizeof(int));
    for (uint i=0; i<n; i++) cover[i] = rand()%256;

    //float* costs = new float[3*n];
    float* costs = (float*)malloc(3*n*sizeof(float));
    for (uint i=0; i<n; i++) {
        if (cover[i]==0) {        // F_INF is defined as infinity in float
            costs[3*i+0] = F_INF; // cost of changing pixel by -1 is infinity => change to -1 is forbidden
            costs[3*i+1] = 0;     // cost of changing pixel by  0
            costs[3*i+2] = 1;     // cost of changing pixel by +1
        } else if (cover[i]==255) {
            costs[3*i+0] = 1;     // cost of changing pixel by -1
            costs[3*i+1] = 0;     // cost of changing pixel by  0
            costs[3*i+2] = F_INF; // cost of changing pixel by +1 is infinity => change to +1 is forbidden
        } else {
            costs[3*i+0] = 1;     // cost of changing pixel by -1
            costs[3*i+1] = 0;     // cost of changing pixel by  0
            costs[3*i+2] = 1;     // cost of changing pixel by +1
        }
    }



    //unsigned char* message = new unsigned char[m];
    unsigned char* message = (unsigned char*)malloc(m*sizeof(unsigned char));

    //unsigned char* extracted_message = new unsigned char[m];
    unsigned char* extracted_message = (unsigned char*)malloc(m*sizeof(unsigned char));

    for (uint i=0; i<m; i++) message[i] = rand()%2;
    
    int* stego = new int[n];
    //unsigned int* num_msg_bits = new unsigned int[2];
    unsigned int* num_msg_bits = (unsigned int*)malloc(2*sizeof(int));

    float coding_loss; // calculate coding loss

    printf("Multi layer construction for steganography.\nExample of weighted +-1 embedding using 2 layers of STCs.\n\n");

    printf("Running stc_pm1_pls_embed()    WITH coding loss calculation ... ");
    double t0 = (double)clock()/CLOCKS_PER_SEC;
    float dist = stc_pm1_pls_embed(n, cover, costs, m, message,
                                   h, F_INF,
                                   stego, num_msg_bits, trials, &coding_loss); // trials contain the number of trials used
    double t = ((double)clock()/CLOCKS_PER_SEC)-t0;
    printf("done in %ld seconds.\n", t);

    trials = 10; // set the maximum number of trials again
    printf("Running stc_pm1_pls_embed() WITHOUT coding loss calculation ... ");
    t0 = (double)clock()/CLOCKS_PER_SEC;
    dist = stc_pm1_pls_embed(n, cover, costs, m, message,
                             h, F_INF,
                             stego, num_msg_bits, trials, 0); // trials contain the number of trials used
    t = ((double)clock()/CLOCKS_PER_SEC)-t0;
    printf("done in %ld  seconds.\n", t);

    printf("Running stc_ml2_extract() ...\n");
    t0 = (double)clock()/CLOCKS_PER_SEC;
    stc_ml_extract(n, stego, 2, num_msg_bits, h, extracted_message);

    printf("done in %ld seconds\n\n", ((double)clock()/CLOCKS_PER_SEC)-t0);

    printf("          Cover size  n = %d elements.\n", n);
    printf("         Message bits m = %d bits => %d bits in 2LSBs and %d bits in LSBs.\n", m, num_msg_bits[1], num_msg_bits[2]);
    printf("STC constraint height h = %d bits\n", h);
    printf("          Coding loss l = %f%\n", coding_loss*100);
    printf("     Processing speed t = %ld cover elements/second without coding loss calculation\n", (double)n/t);
    
    bool msg_ok = true;
    for (uint i=0; i<m; i++) {
        msg_ok &= (extracted_message[i]==message[i]);
        if (!msg_ok) printf("\nExtracted message differs in bit %d\n", i);
    }
    if (msg_ok) printf("\nMessage was embedded and extracted correctly.\n");

    free(cover);
    free(stego);
    free(costs);
    free(message);
    free(extracted_message);
    free(num_msg_bits);

    return 0;
}
