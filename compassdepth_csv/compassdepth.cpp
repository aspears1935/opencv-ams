#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main( int argc, char *argv[] )
{
  char directory[128];
  string value;

  if(argc < 2)
    {
      printf("usage: ./CompassDepth <data-directory>\n");
    }
  
  strcpy(directory, argv[1]);
  string directorytmp(directory);
  
  //cout << directory << "-----" << directorytmp.c_str() << endl;

  //Check for '/' at end of directory string
  if(directory[directorytmp.length()-1] != '/')
    {
      directory[directorytmp.length() + 1] = directory[directorytmp.length()];
      directory[directorytmp.length()] = '/';
    }

  string directorystr(directory);

  //cout << directory << "-----" << directorystr.c_str() << endl;

  //Open Files
  string inFileFrameName = directorystr + string("frames.csv");
  string inFileCompassName = directorystr + string("euler_angles.csv");
  string outFileName = directorystr + string("frames_compass_depth.csv");
  ifstream infileframes(inFileFrameName.c_str());
  ifstream infilecompass(inFileCompassName.c_str());
  ofstream outfile(outFileName.c_str());

  //Initialize output file
  outfile << "timestamp,heading,depth" << endl;

  //Initialize the first compass/depth data
  double time_compass_nsec1;
  double heading1;
  double depth1;
  double time_compass_nsec2;
  double heading2;
  double depth2;
  double time_frames_nsec;
  double time_frames_sec;
  double depth_filt;
  double heading_filt;
  double tmp1, tmp2;

  getline(infilecompass, value, '\n');  //Ignore column headings
  getline(infilecompass, value, ';');
  time_compass_nsec1 = atof(value.c_str());
  getline(infilecompass, value, ';');
  getline(infilecompass, value, ';');
  getline(infilecompass, value, ';');
  heading1 = atof(value.c_str());
  getline(infilecompass, value, ';');
  depth1 = atof(value.c_str());
  //cout << fixed  << time_compass_nsec1 << " " << heading1 << " " << depth1 << endl;
  getline(infilecompass, value, '\n'); //Go to next line
  getline(infilecompass, value, ';');
  time_compass_nsec2 = atof(value.c_str());
  getline(infilecompass, value, ';');
  getline(infilecompass, value, ';');
  getline(infilecompass, value, ';');
  heading2 = atof(value.c_str());
  getline(infilecompass, value, ';');
  depth2 = atof(value.c_str());
  getline(infilecompass, value, '\n'); //Go to the next line
  //cout << fixed << time_compass_nsec2 << " " << heading2 << " " << depth2 << endl;

  while(infileframes.good())
    {
      getline(infileframes, value, ','); //Skip frame number
      getline(infileframes, value, ',');
      time_frames_sec = atof(value.c_str());
      getline(infileframes, value, '\n');
      time_frames_nsec = atof(value.c_str());
      time_frames_nsec += time_frames_sec*1000000000;

      if(time_frames_nsec < time_compass_nsec1); //Frames timestamp < both compass timestamps
	//cout << "C1 " << time_compass_nsec1 << " " << time_frames_nsec << " " << time_compass_nsec2 << endl;

      else if((time_frames_nsec >= time_compass_nsec1)&&(time_frames_nsec <= time_compass_nsec2)); //Frames timestamp between both compass timestamps
	//cout << "C2 " << time_compass_nsec1 << " " << time_frames_nsec << " " << time_compass_nsec2 << endl;

      else if(time_frames_nsec > time_compass_nsec2)
	{
	  while(time_frames_nsec > time_compass_nsec2)
	    {
	      time_compass_nsec1 = time_compass_nsec2;
	      heading1 = heading2;
	      depth1 = depth2;
	      getline(infilecompass, value, ';');
	      time_compass_nsec2 = atof(value.c_str());
	      getline(infilecompass, value, ';');
	      getline(infilecompass, value, ';');
	      getline(infilecompass, value, ';');
	      heading2 = atof(value.c_str());
	      getline(infilecompass, value, ';');
	      depth2 = atof(value.c_str());
	      getline(infilecompass, value, '\n'); //Go to next line
	    }
	  //cout << "C3 " << time_compass_nsec1 << " " << time_frames_nsec << " " << time_compass_nsec2 << endl;
	} 

      depth_filt = (depth1 + depth2)/2;
      heading_filt = (heading1 + heading2)/2;
      if(time_frames_nsec > 1000)  //Fix for end of file bad data
	{
	  //cout << fixed << time_frames_nsec << ',' << heading_filt << ',' << depth_filt << endl;
	  outfile.precision(0);
	  outfile << fixed << time_frames_nsec << ',';
	  outfile.precision(4);
	  outfile << heading_filt << ',' << depth_filt << endl;
	}
    }

  infileframes.close();
  infilecompass.close();
  outfile.close();
  return 0;
}
