#include <bits/stdc++.h>
using namespace std;

double arr[3][3], tnc1, tnc2, tnc3, tc1, tc2, tc3;
double r1, r2, r3, r, p1, p2, p3, p, f, nc1, nc2, nc3, a, t;
int n;

int main(){

	int n;
	cin >> n;
	for(int i = 0 ; i < n ; i++){
		for(int j = 0 ; j < n ; j++){
			cin >> arr[i][j];
			t += arr[i][j];
		}
	}
	nc1 = arr[0][0], nc2 = arr[1][1], nc3 = arr[2][2];

	tc1 = arr[0][0] + arr[0][1] + arr[0][2];
	tc2 = arr[1][0] + arr[1][1] + arr[1][2];
	tc3 = arr[2][0] + arr[2][1] + arr[2][2];

	tnc1 = arr[0][0] + arr[1][0] + arr[2][0];
	tnc2 = arr[0][1] + arr[1][1] + arr[2][1];
	tnc3 = arr[0][2] + arr[1][2] + arr[2][2];

	r1 = nc1 / tc1;
	r2 = nc2 / tc2;

	p1 = nc1 / tnc1;
	p2 = nc2 / tnc2;

	r = (r1 + r2) / 2;
	p = (p1 + p2) / 2;
	f = 2*r*p / (r + p);

	a = (nc1 + nc2 + nc3) / t;

	if(n == 3){
		r3 = nc3 / tc3;
		p3 = nc3 / tnc3;
		r = (r1 + r2 + r3) / 3;
		p = (p1 + p2 + p3) / 3;
		f = 2*r*p / (r + p);

		a = (nc1 + nc2 + nc3) / t;
	}	

	if(n == 2){
		cout << p1 << " " << p2 << endl;
		cout << r1 << " " << r2 << endl;
	}
	else{
		cout << p1 << " " << p2 << " " << p3 << endl;
		cout << r1 << " " << r2 << " " << r3 << endl;
	}
	cout << a << endl;
	cout << f << endl;

	return 0;
}