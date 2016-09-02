#ifndef _GLOBAL_PARAMS_H_
#define _GLOBAL_PARAMS_H_

// size of Build Table R
// Original is 24 Changed as it goes out of memory
#define R_SIZE (1 << 21)

// size of Probe Table S
#define S_SIZE R_SIZE

// size of Input Table (MB)
#define INPUT_SIZE (R_SIZE >> 19)

// maxsize of Result Table T
#define T_MAXSIZE (R_SIZE)

// size of hash table H
// set two times the size of Table R
#define H_SIZE (R_SIZE << 1)

// HASH MASK
#define MASK (H_SIZE - 1)

// min of key in Table R
#define R_KEY_MIN 1

// max of key in Table R use 1U to not have negative numbers
#define R_KEY_MAX (1U << 31)

// match rate of Table S to Table R
#define MATCH_RATE 0.03

// number of streams
#define NS 16

// stream size
#define ST_SIZE (R_SIZE / NS)

// print Table R, S, T or not
//#define PRINT_TABLE

#endif // !_GLOBAL_PARAMS_H_
