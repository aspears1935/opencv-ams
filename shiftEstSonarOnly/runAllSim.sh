./bin/sonar_opencv ../sonar_sim_cloud/circle360.avi >> consoleOut_simCircle360.txt
cp outputSonar.csv outputSonar_simCircle360.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simCircle360.csv

./bin/sonar_opencv ../sonar_sim_cloud/forward200frames1x.avi >> consoleOut_simForward200frames1x.txt
cp outputSonar.csv outputSonar_simForward200frames1x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simForward200frames1x.csv

./bin/sonar_opencv ../sonar_sim_cloud/forward200frames10x.avi >> consoleOut_simForward200frames10x.txt
cp outputSonar.csv outputSonar_forward200frames10x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simForward200frames10x.csv

./bin/sonar_opencv ../sonar_sim_cloud/forward600frames.avi >> consoleOut_simforward600frames.txt
cp outputSonar.csv outputSonar_simForward600frames.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simForward600frames.csv

./bin/sonar_opencv ../sonar_sim_cloud/rect200x100_1x.avi >> consoleOut_simRect200x100_1x.txt
cp outputSonar.csv outputSonar_simRect200x100_1x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simRect200x100_1x.csv

./bin/sonar_opencv ../sonar_sim_cloud/rect200x100_10x.avi >> consoleOut_simRect200x100_10x.txt
cp outputSonar.csv outputSonar_simRect200x100_10x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simRect200x100_10x.csv

./bin/sonar_opencv ../sonar_sim_cloud/scurve_1x.avi 0 0 90 >> consoleOut_simScurve_1x.txt
cp outputSonar.csv outputSonar_simScurve_1x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simScurve_1x.csv

./bin/sonar_opencv ../sonar_sim_cloud/scurve_10x.avi 0 0 90 >> consoleOut_simScurve_10x.txt
cp outputSonar.csv outputSonar_simScurve_10x.csv
cp outputSonarGTSAM.csv outputSonarGTSAM_simScurve_10x.csv