#ifndef COMMON_H
#define COMMON_H

#include <string>

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

extern u32 mats[];

/* Simple class for throwing exceptions */
class stc_exception : public std::exception {
public:
    stc_exception(std::string message, u32 error_id) { this->message = message; this->error_id = error_id; }
    virtual ~stc_exception() throw() {}
    virtual const char* what() const throw() { return message.c_str(); }
    u32 error_id;
private:
    std::string message;
};

/* 
   The following error_ids are in use:
   1 = Submatrix height must not exceed 31.
   2 = Not enough memory.
   3 = The message cannot be longer than the cover object.
   4 = No solution exists.                                 - This happen when there are too many Inf values in cost vector and thus the solution does not exist due to sparse parity-check matrix.
   5 = Price vector limit exceeded.                        - There is a limit to cost elements when you use integer version of the algorithm. Try to use costs in double.
   6 = Maximum number of trials in layered construction exceeded.
 */

u32 *getMatrix(int width, int height);

#endif
