#include <iostream>
#include "CImg.h"
#include <fstream>
#include <random>
#undef Success
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
using namespace cimg_library;

static int label_colors[3*6255];//6255 is the maximum number of segments, but this can be more
static CImgDisplay main_disp;
const unsigned char black[] = { 0,0,0 };

void display_labels(MatrixXi labels){
	//how many different labels are there? -> labels.maxCoeff()
	CImg<unsigned char> image(labels.rows(),labels.cols(),1,3);
	cimg_forXYC(image,x,y,v) { image(x,y,v) = label_colors[(labels(y,x)*3)+v]; }
	//unsigned char purple[] = { 255,0,255 };
	//cimg_forXY(image,x,y) { image(x,y) = purple; }

	main_disp.display(image);

	//while (!main_disp.is_closed())
	//main_disp.wait(5);
}

void display_labels(MatrixXi labels, const char* text){
	//how many different labels are there? -> labels.maxCoeff()
	CImg<unsigned char> image(labels.rows(),labels.cols(),1,3);
	cimg_forXYC(image,x,y,v) { image(x,y,v) = label_colors[(labels(y,x)*3)+v]; }

	image.draw_text(10,10,text,black);
	main_disp.display(image);
}

void display_labels(MatrixXi labels, int n){
	std::string s = std::to_string(n);
	const char *text = s.c_str();  //use char const* as target type
	//how many different labels are there? -> labels.maxCoeff()
	CImg<unsigned char> image(labels.rows(),labels.cols(),1,3);
	cimg_forXYC(image,x,y,v) { image(x,y,v) = label_colors[(labels(y,x)*3)+v]; }

	image.draw_text(10,10,text,black);
	main_disp.display(image);
}

int main(int argc, char** argv)
{
	MatrixXf testm(2,3);
	//cout << testm(0,-1) << endl;
	int a=10;
	int b=10;
	if (a<b-1 && testm(0,0)==testm(0,-1)){
		std::cout << "3" << std::endl;
	}
	//return 1;

	//srand(time(NULL)); /* seed random number generator */
	srand(1);
	//make this many random colors
	for (int i=0;i<3*6255;i++){
		label_colors[i]=rand()%256;
	}

	char* filename;
	char* filename2;
	if( argc != 3)
	{
		cout << "\trequires horizontal .csv file argument and vertical .csv file argument and " << endl;
		return -1;
	}else{
		filename=argv[1];
		filename2=argv[2];
	}
	cout << "reading file " << filename << endl;
	//read the file twice. Once to count elements per line and #lines
	ifstream file (filename); 
	string value, line;
	int rows=0;
	int cols=0;
	int incomplete_rows=0;
	getline (file, line);
	while ( file.good() )
	{
		if(std::count(line.begin(), line.end(), '?')>0)
			incomplete_rows++;
		int elements;
		rows++;
		elements=std::count(line.begin(), line.end(), ',');
		//cout << elements << endl;
		if(cols==0)
			cols=elements;
		else if(elements>0 && elements!=cols){
			cout << "bad input file, unequal rows (misplaced comma?)" << endl;
			return -1;
		}
		getline (file, line);
	}
	cols++;
	cout << "horizontal file has " << rows << " rows with " << cols << " columns each, and " << incomplete_rows << " incomplete rows" << endl;

	file.clear();
	//and the second time to put everything into a matrix
	MatrixXf csvh(rows-incomplete_rows,cols);
	file.seekg(0, ios::beg);

	std::string token;
	int i=0;
	int incomplete_i=0;
	getline (file, line);
	while ( file.good() )
	{
		std::istringstream ss(line);
		if(std::count(line.begin(), line.end(), '?')>0)//contains '?'
			std::cout << "incomplete line in horizontal file" << endl;
		else{
			for(int j=0;j<cols;j++){
				std::getline(ss, token, ',');
				//std::cout << "(" << j << ", " << i << "):" << token << '\n';
				csvh(i,j) = atof(token.c_str());
			}
			++i;
		}
		getline (file, line);
	}
	//std::cout << "horizontal from file is:\n" << csvh << endl;

	cout << "reading file " << filename2 << endl;
	//read the file twice. Once to count elements per line and #lines
	ifstream file2 (filename2);
	rows=0;
	cols=0;
	incomplete_rows=0;
	getline (file2, line);
	while ( file2.good() )
	{
		if(std::count(line.begin(), line.end(), '?')>0)
			incomplete_rows++;
		int elements;
		rows++;
		elements=std::count(line.begin(), line.end(), ',');
		//cout << elements << endl;
		if(cols==0)
			cols=elements;
		else if(elements>0 && elements!=cols){
			cout << "bad input file2, unequal rows (misplaced comma?)" << endl;
			return -1;
		}
		getline (file2, line);
	}
	cols++;
	cout << "vertical file has " << rows << " rows with " << cols << " columns each, and " << incomplete_rows << " incomplete rows" << endl;

	file2.clear();
	//and the second time to put everything into a matrix
	MatrixXf csvv(rows-incomplete_rows,cols);
	file2.seekg(0, ios::beg);

	i=0;
	incomplete_i=0;
	getline (file2, line);
	while ( file2.good() )
	{
		std::istringstream ss(line);
		if(std::count(line.begin(), line.end(), '?')>0)//contains '?'
			std::cout << "incomplete line in vertical file" << endl;
		else{
			for(int j=0;j<cols;j++){
				std::getline(ss, token, ',');
				//std::cout << "(" << j << ", " << i << "):" << token << '\n';
				csvv(i,j) = atof(token.c_str());
			}
			++i;
		}
		getline (file2, line);
	}
	//std::cout << "horizontal from file2 is:\n" << csvv << endl;


	if(csvh.rows() != csvh.cols()+1 || csvh.rows() != csvv.cols() || csvv.rows()+1 != csvv.cols()){
		cout << "matrices don't have the expected size (square +-1)" << endl;
		cout << "horizontal rows: " << csvh.rows() << ", cols: " << csvh.cols() << endl;
		cout << "vertical rows: " << csvv.rows() << ", cols: " << csvv.cols() << endl;
		return -1;
	}
	rows=csvv.cols();
	cols=csvh.rows();

	//generate random labels
	int max_label=7;
	MatrixXi labels(rows,cols);
	for (int i=0;i<labels.cols();i++)
		for (int j=0;j<labels.rows();j++)
			labels(i,j)=rand()%max_label;//start with 15 partitions all around

	MatrixXf gibbs(rows,cols);//get the probabilities for changing it
	for (int n=0;n<10000;n++){
		for (int i=1;i<cols-1;i++){
			for (int j=1;j<rows-1;j++){
				gibbs(i,j)=1;
				if(labels(i,j)==labels(i,j+1))
					gibbs(i,j)*=(csvv(i,j));
				else
					gibbs(i,j)*=(1-csvv(i,j));
				if(labels(i,j)==labels(i,j-1))
					gibbs(i,j)*=(csvv(i,j-1));
				else
					gibbs(i,j)*=(1-csvv(i,j-1));
				if(labels(i,j)==labels(i+1,j))
					gibbs(i,j)*=(csvh(i,j));
				else
					gibbs(i,j)*=(1-csvh(i,j));
				if(labels(i,j)==labels(i-1,j))
					gibbs(i,j)*=(csvh(i-1,j));
				else
					gibbs(i,j)*=(1-csvh(i-1,j));
			}
		}
		//gibbs probabilities for top and bottom rows
		for (int i=0;i<cols;i++){
			gibbs(0,i)=1;
			if(labels(0,i)==labels(1,i))
				gibbs(0,i)*=(csvv(0,i));
			else
				gibbs(0,i)*=(1-csvv(0,i));

			if(i<cols-1)
				if(labels(0,i)==labels(0,i+1))
					gibbs(0,i)*=(csvh(0,i));
				else
					gibbs(0,i)*=(1-csvh(0,i));
			if(i>0)
				if(labels(0,i)==labels(0,i-1))
					gibbs(0,i)*=(csvh(0,i-1));
				else
					gibbs(0,i)*=(1-csvh(0,i-1));

			gibbs(rows-1,i)=1;			
			if(labels(rows-1,i)==labels(rows-2,i)){
			
				gibbs(rows-1,i)*=(csvv(rows-2,i));
			}else{
			
				gibbs(rows-1,i)*=(1-csvv(rows-2,i));
			}

			if(i<cols-1)
				if(labels(rows-1,i)==labels(rows-1,i+1))
					gibbs(rows-1,i)*=(csvh(rows-1,i));
				else
					gibbs(rows-1,i)*=(1-csvh(rows-1,i));
			if(i>0)
				if(labels(rows-1,i)==labels(rows-1,i-1))
					gibbs(rows-1,i)*=(csvh(rows-1,i-1));
				else
					gibbs(rows-1,i)*=(1-csvh(rows-1,i-1));
		}
		//gibbs probabilities for left and right collumns
		for (int i=0;i<rows;i++){
			gibbs(i,0)=1;
			if(labels(i,0)==labels(i,1))
				gibbs(i,0)*=(csvh(i,0));
			else
				gibbs(i,0)*=(1-csvh(i,0));

			if(i<rows-1)
				if(labels(i,0)==labels(i+1,0))
					gibbs(i,0)*=(csvv(i,0));
				else
					gibbs(i,0)*=(1-csvv(i,0));
			if(i>0)
				if(labels(i,0)==labels(i-1,0))
					gibbs(i,0)*=(csvv(i-1,0));
				else
					gibbs(i,0)*=(1-csvv(i-1,0));

			gibbs(i,cols-1)=1;
			if(labels(i,cols-1)==labels(i,cols-2))
				gibbs(i,cols-1)*=(csvh(i,cols-2));
			else
				gibbs(i,cols-1)*=(1-csvh(i,cols-2));

			if(i<rows-1)
				if(labels(i,cols-1)==labels(i+1,cols-1))
					gibbs(i,cols-1)*=(csvv(i,cols-1));
				else
					gibbs(i,cols-1)*=(1-csvv(i,cols-1));
			if(i>0)
				if(labels(i,cols-1)==labels(i-1,cols-1))
					gibbs(i,cols-1)*=(csvv(i-1,cols-1));
				else
					gibbs(i,cols-1)*=(1-csvv(i-1,cols-1));
		}
		//resample
		for (int i=0;i<cols;i++)
			for (int j=0;j<rows;j++)
				if(gibbs(i,j)>((double) rand() / (RAND_MAX)))//resample
					labels(i,j)=rand()%max_label;		
		if(n%10==0)
			display_labels(labels,n);
	}
	cout << "Labels:\n" << labels << endl;
	return 1;
}
