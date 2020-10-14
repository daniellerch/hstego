#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

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
    int* cover = new int[n];
    for (uint i=0; i<n; i++) cover[i] = rand()%256;

    float* costs = new float[5*n];
    for (uint i=0; i<n; i++) {
        for (int j=-2; j<3; j++) {
            costs[5*i+2+j] = F_INF; // default value
            if ( (cover[i]+j>=0) && (cover[i]+j<=255) ) {
                switch (j) {
                case -2:
                case 2:
                    costs[5*i+2+j] = 4; // cost of changing pixel by  +-2
                    break;
                case -1:
                case 1:
                    costs[5*i+2+j] = 1; // cost of changing pixel by  +-1
                    break;
                default:
                    costs[5*i+2+j] = 0; // cost of changing pixel by  0
                }
            }
        }
    }

    unsigned char* message = new unsigned char[m];
    unsigned char* extracted_message = new unsigned char[m];
    for (uint i=0; i<m; i++) message[i] = rand()%2;
    
    int* stego = new int[n];
    unsigned int* num_msg_bits = new unsigned int[3];
    float coding_loss; // calculate coding loss

    std::cout << "Multi layer construction for steganography.\nExample of weighted +-2 embedding using 3 layers of STCs in payload limited regime.\n\n";

    std::cout << "Running stc_pm2_pls_embed()    WITH coding loss calculation ... " << std::flush;
    double t0 = (double)clock()/CLOCKS_PER_SEC;
    float dist = stc_pm2_pls_embed(n, cover, costs, m, message,
                                   h, F_INF,
                                   stego, num_msg_bits, trials, &coding_loss); // trials contain the number of trials used
    double t = ((double)clock()/CLOCKS_PER_SEC)-t0;
    std::cout << "done in " << t << " seconds.\n";

    trials = 10; // set the maximum number of trials again
    std::cout << "Running stc_pm2_pls_embed() WITHOUT coding loss calculation ... " << std::flush;
    t0 = (double)clock()/CLOCKS_PER_SEC;
    dist = stc_pm2_pls_embed(n, cover, costs, m, message,
                                   h, F_INF,
                             stego, num_msg_bits, trials, 0); // trials contain the number of trials used
    t = ((double)clock()/CLOCKS_PER_SEC)-t0;
    std::cout << "done in " << t << " seconds.\n";

    std::cout << "Running stc_ml3_extract() ... " << std::flush;
    t0 = (double)clock()/CLOCKS_PER_SEC;
    stc_ml_extract(n, stego, 3, num_msg_bits, h, extracted_message);
    std::cout << "done in " << ((double)clock()/CLOCKS_PER_SEC)-t0 << " seconds.\n\n";

    std::cout << "          Cover size  n = " << n << " elements.\n";
    std::cout << "         Message bits m = " << m << " bits => " << num_msg_bits[2] << " bits in 3LSBs, " << num_msg_bits[1] << " bits in 2LSBs and " << num_msg_bits[0] << " bits in LSBs.\n";
    std::cout << "STC constraint height h = " << h << " bits\n";
    std::cout << "          Coding loss l = " << coding_loss*100 << "%.\n";
    std::cout << "     Processing speed t = " << (double)n/t << " cover elements/second without coding loss calculation\n";
    
    bool msg_ok = true;
    for (uint i=0; i<m; i++) {
        msg_ok &= (extracted_message[i]==message[i]);
        if (!msg_ok) { std::cout << "\nExtracted message differs in bit " << i << std::endl; break; }
    }
    if (msg_ok) std::cout << "\nMessage was embedded and extracted correctly." << std::endl;

    delete[] cover;
    delete[] stego;
    delete[] costs;
    delete[] message;
    delete[] extracted_message;
    delete[] num_msg_bits;

    return 0;
}
