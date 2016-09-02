#ifndef _TABLE_CUH_
#define _TABLE_CUH_

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "global.h"

// CUDA
#define CUDACheckError(error) { cudaCheckError((error), __FILE__, __LINE__); }
//#define CUDACheckError(error) { error; }

inline void cudaCheckError(cudaError_t error, const char *file, int line)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR (%d) {%s} file: %s %d\n", error, cudaGetErrorString(error), file, line);
		//Windows OS specific
		//system("pause");
		exit(1);
	}
}

// TableR
class TupleR
{
public:
	uint32_t r_id;
	uint32_t key;

public:
	__host__ __device__ TupleR() :r_id(0), key(0){}
	__host__ __device__ TupleR(uint32_t r, uint32_t k) : r_id(r), key(k){}

};

// TableS
class TupleS
{
public:
	uint32_t s_id;
	uint32_t key;

public:
	__host__ __device__ TupleS() :s_id(0), key(0){}
	__host__ __device__ TupleS(uint32_t s, uint32_t k) : s_id(s), key(k){}

};

// TableT
class TupleT
{
public:
	uint32_t r_id;
	uint32_t s_id;

public:
	__host__ __device__ TupleT() :r_id(0), s_id(0) {}
	__host__ __device__ TupleT(uint32_t r, uint32_t s) : r_id(r), s_id(s) {}

};

// HashTable
typedef TupleR TupleH;

// TableR
class TableR
{
public:
	TupleR *pr;

public:
	TableR() :pr(NULL){}

	void init();
	void print();
	void clear();
};

// TableS
class TableS
{
public:
	TupleS *ps;

public:
	TableS() :ps(NULL){}

	void init(TableR *r);
	void print();
	void clear();
};

// TableT
class TableT
{
public:
	TupleT *pt;
	uint32_t size;

public:
	TableT() :pt(NULL), size(0){}

	void init();
	void init(uint32_t t_size);
	void pushback(TupleT tuple);
	void check(TableR *r, TableS *s);
	void print();
	void clear();
};

// HashTable
class HashTable
{
public:
	TupleH *ph;

public:
	HashTable() :ph(NULL){}

	void init();
	void build(TableR *r);
	void print();
	void probe(TableS *s, TableT *t);
};


// least significant bits of the input key as hash function
__host__ __device__ inline uint32_t LSB(uint32_t key)
{
	return key & MASK; // 与H_SIZE有关
}

// init TableR
void TableR::init()
{
	pr = (TupleR *)malloc(R_SIZE * sizeof(TupleR));


	time_t seed;
	time(&seed);
	//fprintf(stdout, "time seed is %u \n", seed);


	std::uniform_int_distribution<uint32_t> u(R_KEY_MIN, R_KEY_MAX);
	std::default_random_engine e((uint32_t)seed);

	for (uint32_t i = 0; i < R_SIZE; i++)
	{
		pr[i].r_id = i;
		pr[i].key = u(e);
	}
}

// print TableR
void TableR::print()
{
	fprintf(stdout, "Table R size is %u\n", R_SIZE);
	fprintf(stdout, "Table R: \n");

#ifdef PRINT_TABLE
	for (uint32_t i = 0; i < R_SIZE; i++)
	{
		fprintf(stdout, "%u %u \n", pr[i].r_id, pr[i].key);
	}
#endif
}

// clear TableR
void TableR::clear()
{
	free(pr);
	pr = NULL;
}

// init TableS
void TableS::init(TableR *r)
{
	ps = (TupleS *)malloc(S_SIZE * sizeof(TupleS));

	time_t seed;
	time(&seed);
	//fprintf(stdout, "time seed is %u \n", seed);

	// match rate
	std::uniform_real_distribution<double> u1(0.0, 1.0);
	std::default_random_engine e1((uint32_t)(seed + 10));

	// Table S
	std::uniform_int_distribution<uint32_t> u2(0, R_SIZE - 1);
	std::default_random_engine e2((uint32_t)(seed + 100));

	// key
	std::uniform_int_distribution<uint32_t> u3(R_KEY_MAX + 1, UINT32_MAX);
	std::default_random_engine e3((uint32_t)(seed + 1000));

	for (uint32_t i = 0; i < S_SIZE; i++)
	{
		ps[i].s_id = i;

		if (u1(e1) < MATCH_RATE)
		{
			ps[i].key = r->pr[u2(e2)].key;
		}
		else
		{
			//ps[i].key = 0; //不匹配全部设为0， 此处可能不合理
			ps[i].key = u3(e3);
		}
	}
}

// print TableS
void TableS::print()
{
	fprintf(stdout, "Table S size is %u\n", S_SIZE);
	fprintf(stdout, "Table S: \n");

#ifdef PRINT_TABLE
	for (uint32_t i = 0; i < S_SIZE; i++)
	{
		fprintf(stdout, "%u %u \n", ps[i].s_id, ps[i].key);
	}
#endif
}

// clear TableS
void TableS::clear()
{
	free(ps);
	ps = NULL;
}

// init TableT by T_MAXSIZE
void TableT::init()
{
	size = 0;
	pt = (TupleT *)malloc(T_MAXSIZE * sizeof(TupleT));
}

// init TableT by t_size
void TableT::init(uint32_t t_size)
{
	size = 0;
	pt = (TupleT *)malloc(t_size * sizeof(TupleT));
}

void TableT::pushback(TupleT tuple)
{
	if (size == T_MAXSIZE)
	{
		fprintf(stdout, "TableT is full!");
	}
	pt[size] = tuple;
	size++;
}

// check TableT
void TableT::check(TableR *r, TableS *s)
{
	for (uint32_t i = 0; i < size; i++)
	{
		if (r->pr[pt[i].r_id].key != s->ps[pt[i].s_id].key)
		{
			fprintf(stdout, "r_id = %u, r_key = %u, s_id = %u, s_key = %u \n", pt[i].r_id, r->pr[pt[i].r_id].key, pt[i].s_id, s->ps[pt[i].s_id].key);
			fprintf(stdout, "TableT is wrong!\n");
			return;
		}
	}
	fprintf(stdout, "TableT is right!\n");
}

// print TableT
void TableT::print()
{
	fprintf(stdout, "Table T size is %u\n", size);
	fprintf(stdout, "Table T: \n");

#ifdef PRINT_TABLE
	for (uint32_t i = 0; i < size; i++)
	{
		fprintf(stdout, "%u %u \n", pt[i].r_id, pt[i].s_id);
	}
#endif
}

// clear TableT
void TableT::clear()
{
	size = 0;
	free(pt);
	pt = NULL;
}

//////////////////////

// init HashTable
void HashTable::init()
{
	ph = (TupleH *)malloc(H_SIZE * sizeof(TupleH));

	for (uint32_t i = 0; i < H_SIZE; i++)
	{
		ph[i] = TupleH(0, 0);
	}
}

// build HashTable
void HashTable::build(TableR *r)
{
	for (uint32_t i = 0; i < R_SIZE; i++)
	{
		uint32_t hk = LSB(r->pr[i].key);
		while (ph[hk].key != 0)
		{
			hk = (hk + 1) % H_SIZE;  //线性探测
		}

		ph[hk] = r->pr[i];
	}
}

// print HashTable
void HashTable::print()
{
	fprintf(stdout, "HashTable H size is %u\n", H_SIZE);
	fprintf(stdout, "HashTable H: \n");

#ifdef PRINT_TABLE
	for (uint32_t i = 0; i < H_SIZE; i++)
	{
		fprintf(stdout, "%u %u \n", ph[i].r_id, ph[i].key);
	}
#endif
}

void HashTable::probe(TableS *s, TableT *t)
{
	for (uint32_t i = 0; i < S_SIZE; i++)
	{
		uint32_t hk = LSB(s->ps[i].key);

		while (ph[hk].key != 0)
		{
			if (ph[hk].key == s->ps[i].key)
			{
				t->pushback(TupleT(ph[hk].r_id, s->ps[i].s_id));
			}
			hk = (hk + 1) % H_SIZE;
		}
	}
}

#endif
