
#include"mis.h"

void MISLPSolution::save(path filename) {
	ofstream file;
	file.open(filename);
	// variable features
	for(size_t i=0;i<x.size();i++) {
		file << x[i] << "," << cost[i] << "," << reduced_cost[i] << endl;
	}
	// constraint features
	for(size_t i=0;i<slack.size();i++) {
		file << slack[i] << "," << rhs[i] << "," << dual[i] << endl;
	}
	file.close();
}


struct MISLPSolution cplex_solve_relaxed_mis(Graph& g) {
	int n = num_vertices(g), m=num_edges(g);
	IloEnv myenv;
	IloModel mymodel(myenv);
	// variables, one for each vertex
	IloNumVarArray x(myenv, n, 0, 1, ILOFLOAT);
	// add a constraint for each edge of the graph
	// get the property map for vertex indices
	IndexMap index = get(vertex_index, g);
	graph_traits<Graph>::edge_iterator ei, ei_end;
	int u,v;
	IloRangeArray constraints(myenv);
    	for(boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei) {
		u = index[source(*ei,g)];
		v = index[target(*ei,g)];
		constraints.add(IloRange(myenv, -IloInfinity, x[u] + x[v] , 1));
	}
	mymodel.add(constraints);
	// add objective
	IloExpr myexpr(myenv); // empty expression
	for(int j=0;j<n;j++) {
		myexpr += x[j];
	}
	mymodel.add(IloMaximize(myenv, myexpr));
	myexpr.end();
	// solve
	IloCplex mycplex(myenv);
	mycplex.extract(mymodel);
	mycplex.setOut(myenv.getNullStream());
	cout << "Solving..." << flush;
	mycplex.solve();
	cout << "done" << endl << flush;
	struct MISLPSolution sol;
	if(mycplex.getStatus()==IloAlgorithm::Status::Optimal) {
		//cout << "Objective values" << endl;
		//cout << "\tGreedy = " << greedy_sol << endl;
		for(int i=0;i<n;i++) {
			sol.x.push_back(mycplex.getValue(x[i]));
			sol.cost.push_back(1.);
			sol.reduced_cost.push_back(mycplex.getReducedCost(x[i]));
		}
		for(int i=0;i<m;i++) {
			sol.slack.push_back(mycplex.getSlack(constraints[i]));
			sol.rhs.push_back(1.);
			sol.dual.push_back(mycplex.getDual(constraints[i]));
		}
	}
	mycplex.clear();
	myenv.end();
	return sol;
}

struct MISLPSolution cplex_solve_relaxed_mis(METISGraph& g) {
	int n = g.nv(), m=g.ne();
	IloEnv myenv;
	IloModel mymodel(myenv);
	// variables, one for each vertex
	IloNumVarArray x(myenv, n, 0, 1, ILOFLOAT);
	// add a constraint for each edge of the graph
	// get the property map for vertex indices
	vector<std::pair<int,int> > edge_l = g.edges();
	int u,v;
	IloRangeArray constraints(myenv);
    	for(std::pair<int,int>& e : edge_l) {
		u = e.first;
		v = e.second;
		constraints.add(IloRange(myenv, -IloInfinity, x[u] + x[v] , 1));
	}
	mymodel.add(constraints);
	// add objective
	IloExpr myexpr(myenv); // empty expression
	for(int j=0;j<n;j++) {
		myexpr += x[j];
	}
	mymodel.add(IloMaximize(myenv, myexpr));
	myexpr.end();
	// solve
	IloCplex mycplex(myenv);
	mycplex.extract(mymodel);
	mycplex.setOut(myenv.getNullStream());
	cout << "Solving..." << flush;
	mycplex.solve();
	cout << "done" << endl << flush;
	struct MISLPSolution sol;
	if(mycplex.getStatus()==IloAlgorithm::Status::Optimal) {
		//cout << "Objective values" << endl;
		//cout << "\tGreedy = " << greedy_sol << endl;
		for(int i=0;i<n;i++) {
			sol.x.push_back(mycplex.getValue(x[i]));
			sol.cost.push_back(1.);
			sol.reduced_cost.push_back(mycplex.getReducedCost(x[i]));
		}
		for(int i=0;i<m;i++) {
			sol.slack.push_back(mycplex.getSlack(constraints[i]));
			sol.rhs.push_back(1.);
			sol.dual.push_back(mycplex.getDual(constraints[i]));
		}
	}
	mycplex.clear();
	myenv.end();
	return sol;
}


void cplex_solve_mis(Graph& g) {
	int n = num_vertices(g);
	IloEnv myenv;
	IloModel mymodel(myenv);
	// variables, one for each vertex
	IloNumVarArray x(myenv, n, 0, 1, ILOBOOL);
	// add a constraint for each edge of the graph
	// get the property map for vertex indices
	IndexMap index = get(vertex_index, g);
	graph_traits<Graph>::edge_iterator ei, ei_end;
	int u,v;
	IloRangeArray constraints(myenv);
    	for(boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei) {
		u = index[source(*ei,g)];
		v = index[target(*ei,g)];
		constraints.add(IloRange(myenv, -IloInfinity, x[u] + x[v] , 1));
	}
	mymodel.add(constraints);
	// add objective
	IloExpr myexpr(myenv); // empty expression
	for(int j=0;j<n;j++) {
		myexpr += x[j];
	}
	mymodel.add(IloMaximize(myenv, myexpr));
	myexpr.end();
	// solve
	IloCplex mycplex(myenv);
	mycplex.extract(mymodel);
	mycplex.setOut(myenv.getNullStream());
	mycplex.solve();
	if(mycplex.getStatus()==IloAlgorithm::Status::Optimal) {
		cout << "Objective values" << endl;
		cout << "x = [";
		for(int i=0;i<n;i++) {
			cout << mycplex.getValue(x[i]) << " ";
		}
		cout << "]" << endl;
	}
	mycplex.clear();
	myenv.end();
}

