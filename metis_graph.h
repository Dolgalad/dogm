#ifndef METIS_GRAPH_H
#define METIS_GRAPH_H

#include<vector>
#include<fstream>
#include<algorithm>

#include"string_utils.h"

using namespace std;

struct EdgeFinder {
	std::pair<int,int> edge;
	EdgeFinder(std::pair<int,int> e) : edge(e) {};
	bool operator()(std::pair<int,int>& e) {
		return (e.first==edge.first && e.second==edge.second) || (e.second==edge.first && e.first==edge.second);
	}
	bool operator==(std::pair<int,int>& e) {
		return (e.first==edge.first && e.second==edge.second) || (e.second==edge.first && e.first==edge.second);
	}

};

class METISGraph {
	public:
		METISGraph() : n(0), m(0), fmt(0), ncon(0) {};
		METISGraph(unsigned long _n, unsigned long _m, unsigned int _fmt, unsigned int _ncon, vector<vector<unsigned long> >& _vnl, vector<vector<unsigned long> >& _ewl, vector<vector<unsigned long> >& _vwl) : n(_n), m(_m), fmt(_fmt), ncon(_ncon), vertex_neighbors(_vnl), edge_weights(_ewl), vertex_weights(_vwl)  {};

		unsigned long nv() {return n;};
		unsigned long ne() {return m;};
		vector<unsigned long> neighbors(int i) {
			return vertex_neighbors[i];
		};

		vector<std::pair<int,int> > edges() {
			vector<std::pair<int,int> > edge_l;
			for(unsigned long i=0;i<n;i++) {
				for(unsigned int j=0;j<vertex_neighbors[i].size();j++) {
					std::pair<int,int> edge(i,vertex_neighbors[i][j]);
					if(find_if(edge_l.begin(), edge_l.end(), EdgeFinder(edge))==edge_l.end()) {
						edge_l.push_back(edge);
					}
				}
			}
			return edge_l;
		};

		static METISGraph load(string filename) {
			METISGraph g;
			// open file
			string line;
			int line_c = 0;
			ifstream file(filename);
			if(file.is_open()) {
				while(getline(file, line)) {
					if(line.rfind("%") == 0) {
						continue;
					}
					vector<string> tokens = split_string(line, " ");
					if(line_c == 0) {
						// first line contains number of vertices and edges
						g.n = stoi(tokens[0]);
						g.m = stoi(tokens[1]);
						if(tokens.size()>2) g.fmt = stoi(tokens[2]);
						if(g.fmt == 10 || g.fmt==11) g.ncon = 1;
						if(tokens.size()>3) g.ncon = stoi(tokens[3]);
					} else {
						vector<unsigned long> vw, vn, ew;
						for(unsigned int i=0;i<g.ncon;i++) vw.push_back(stoi(tokens[i]));
						int step = (g.fmt==0 || g.fmt==10) ? 1 : 2;
						for(unsigned int i=g.ncon;i<tokens.size();i+=step) {
							if(g.fmt==0 || g.fmt==10) {
								// there are no edge weights
								vn.push_back(stoi(tokens[i])-1);
							} else {
								// there are edge weights
								ew.push_back(stoi(tokens[i+1]));
								vn.push_back(stoi(tokens[i])-1);
							}
						}
						g.vertex_neighbors.push_back(vn);
						g.vertex_weights.push_back(vw);
						if(g.fmt==1 || g.fmt==11) {
							g.edge_weights.push_back(ew);
						}
					}

					line_c++;
				}
			}
			file.close();
			return g;
		};
		void save(std::string filename) {
			ofstream outfile;
			outfile.open(filename);
			// write the first line : number of vertices, number of edges, feature format, vertex feature count
			outfile << n << " " << m;
			if(fmt>0) {
				outfile << " " << fmt;
				if((fmt==10 || fmt==11) && ncon>1) {
					outfile << " " << ncon;
				}
			}
			outfile << endl;
			// vertex and edge data
			uint64_t i,j;
			for(i=0;i<n;i++) {
				for(j=0;j<ncon;j++) {
					if(j==0) outfile << vertex_weights[i][j];
					else outfile << " " << vertex_weights[i][j];
				}
				for(j=0;j<vertex_neighbors[i].size();j++) {
					if(ncon==0 && j==0) {
						outfile << vertex_neighbors[i][j]+1;
					} else {
						outfile << " " << vertex_neighbors[i][j]+1;
					}
					if(fmt==1 || fmt==11) {
						outfile << " " << edge_weights[i][j];
					}
				}
				outfile << endl;
			}
			outfile.close();
		};
		void show() {
			cout << "n = " << n << endl;
			cout << "m = " << m << endl;
			cout << "fmt = " << fmt << endl;
			cout << "ncon = " << ncon << endl;
			cout << vertex_weights.size() << ", " << vertex_neighbors.size() << ", " << edge_weights.size() << endl;
			for(unsigned int i=0;i<n;i++) {
				cout << "Vertex " << i;
				if(ncon) {
					cout << " : [";
					for(auto v : this->vertex_weights[i]) cout << v << " ";
					cout << "]";
				}
				cout << endl;
				// neighbors
				for(size_t j=0;j<vertex_neighbors[i].size();j++) {
					cout << "\t-> " << vertex_neighbors[i][j];
					if(fmt==1 || fmt==11) cout << " [" << edge_weights[i][j] << "]";
					cout << endl;
				}
			}
		};
	private:
		unsigned long n,m;
		unsigned int fmt,ncon;
		vector<int> src,dst;
		vector<vector<unsigned long> > vertex_neighbors, edge_weights; 
		vector<vector<unsigned long> > vertex_weights;
};

#endif
