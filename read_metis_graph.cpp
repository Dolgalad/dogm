#include<iostream>
#include<fstream>
#include<string>

#include"metis_graph.h"

using namespace std;

int main(int argc, char ** argv) {
	if(argc <= 1) {
		cout << "Please supply a filename" << endl;
	}
	string filename = argv[1];
	cout << "Reading graph from '"<< filename << endl;
	METISGraph g = METISGraph::load(filename);
	cout << "Number of vertices : " << g.nv() << endl;
	cout << "Number of edges    : " << g.ne() << endl;

	g.show();

	g.save("metis_graph_test.graph");

	return 0;
}
