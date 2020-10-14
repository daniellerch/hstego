#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "stc_ml_c.h"

int main() {
    
    for(int it=50; it>0; it-=1) {

        float payload = (float)it/100;

        const uint n = 512*512;    // cover size
        uint h = 10;               // constraint height of STC code
        uint m = floor(payload*n); // number of message bits to embed
        uint trials = 10;          // if the message cannot be embedded due to large amount of 
                                   // wet pixels, then try again with smaller message. Try at most 10 times.
        
        srand(0);
        // srand(time(NULL));

        // generate random 8bit cover
        int* cover = new int[n];
        for (uint i=0; i<n; i++) cover[i] = rand()%256;

        float* costs = new float[3*n];
        for (uint i=0; i<n; i++) {
            if (cover[i]==0) {        // F_INF is defined as infinity in float
                costs[3*i+0] = F_INF; // cost of changing pixel by -1 is infinity => change to -1 is forbidden
                costs[3*i+1] = 0;     // cost of changing pixel by  0
                //costs[3*i+2] = 1;     // cost of changing pixel by +1
                costs[3*i+2] = rand()%100;     // cost of changing pixel by +1
            } else if (cover[i]==255) {
                //costs[3*i+0] = 1;     // cost of changing pixel by -1
                costs[3*i+0] = rand()%100;     // cost of changing pixel by -1
                costs[3*i+1] = 0;     // cost of changing pixel by  0
                costs[3*i+2] = F_INF; // cost of changing pixel by +1 is infinity => change to +1 is forbidden
            } else {
                //costs[3*i+0] = 1;     // cost of changing pixel by -1
                costs[3*i+0] = rand()%100;     // cost of changing pixel by -1
                costs[3*i+1] = 0;     // cost of changing pixel by  0
                //costs[3*i+2] = 1;     // cost of changing pixel by +1
                costs[3*i+2] = rand()%100;     // cost of changing pixel by +1
            }
        }

        unsigned char* message = new unsigned char[m];
        unsigned char* extracted_message = new unsigned char[m];
        for (uint i=0; i<m; i++) message[i] = rand()%2;
        
        int* stego = new int[n];
        unsigned int* num_msg_bits = new unsigned int[2];
        float coding_loss; // calculate coding loss

        float dist = stc_pm1_pls_embed(n, cover, costs, m, message, h, F_INF,
                                   stego, num_msg_bits, trials, &coding_loss); 

        trials = 7;
        dist = stc_pm1_pls_embed(n, cover, costs, m, message, h, F_INF, stego, 
                                 num_msg_bits, trials, 0);

        std::cout << "Payload: " << payload << ", num_msg_bits = " << num_msg_bits[0] << ", " << num_msg_bits[1] << std::endl;

        stc_ml_extract(n, stego, 2, num_msg_bits, h, extracted_message);

        bool msg_ok = true;
        for (uint i=0; i<m; i++) {
            msg_ok &= (extracted_message[i]==message[i]);
            if (!msg_ok) {
                std::cout << "ERROR!, position: " << i << std::endl;
                break;
            }
        }
        //if (msg_ok) std::cout << "OK" << std::endl;

        delete[] cover;
        delete[] stego;
        delete[] costs;
        delete[] message;
        delete[] extracted_message;
        delete[] num_msg_bits;
    }

    return 0;
}
