#ifndef BELIEF_PROPAGATION_H
#define BELIEF_PROPAGATION_H

#include<iostream>
#include<cmath>
#include<vector>
#include<chrono>

#include<torch/torch.h>

#include"metis_graph.h"

using namespace std;
using namespace torch;
using namespace torch::indexing;

vector<bool> value_to_bool_vector(unsigned int val, int size);
Tensor scatter_logsumexp(const Tensor& idx, const Tensor& src, float eps, int idx_n);


struct FactorGraph {
	long int n, nF;
	vector<vector<long int> > factor_neighbors, variable_neighbors;

	FactorGraph() {};
	FactorGraph(const FactorGraph& fg);

	long int num_sufficient_statistics();
	long int num_f2v_messages();
	long int num_f2v_edges();
	long int num_r_edges();
	long int num_q_edges();
	long int num_m_edges();
	long int num_vm_edges();
	vector<long int> factor_value_sizes();

	vector<long int> factor_value_offsets();
	vector<long int> factor_sizes();
	unsigned int factor_size(int i);
};

struct FGEdges {
	Tensor r_edges, q_edges, f2v_edges, m_edges, vm_edges;
	Tensor num_vars, num_ss, num_fact, num_f2v_msg;
	Tensor m_stride;
	Tensor ss_index, v_ss_index_mask;

	FGEdges();
	FGEdges(const struct FGEdges& fe);
	void to(torch::Device device, bool pin_memory);

	void make_edges(struct FactorGraph& fg);
};

Tensor sum_product_loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, int max_iter, double tol);
Tensor loopy_belief_propagation(Tensor& theta, Tensor& q, FGEdges& edges, FactorGraph& fg, string mode, int max_iter, double tol);

#endif
