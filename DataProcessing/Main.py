

from DatasetCreator import DatasetCreator
import sys
sys.path.append("/home/hanna/Documents/ETH/masterthesis/FrontNetPorting/DataProcessing")
import config


def main():
	# subject_name = "davide1"
	# rosbagName = config.folder_path + "/data/Hand/" + subject_name + ".bag"
	# pickleName = config.folder_path + "/data/Hand/" + subject_name + "Hand.pickle"
	# CreateDatasetFromRosbag(rosbagName, pickleName, isBebop=True, start_frame=None, end_frame=None)

	subject_name = "16"
	rosbagName = config.folder_path + "/data/" + subject_name + ".bag"
	pickleName = config.folder_path + "/data/" + subject_name + ".pickle"
	print(rosbagName)
	#CreateDatasetFromDarioRosbag(rosbagName, pickleName, start_frame=None, end_frame=None)

	dc = DatasetCreator(rosbagName)
	dc.CreateHimaxDataset(0, pickleName, None, "/bebop/image_raw/compressed", "/optitrack/bebop", ["/optitrack/head"])
	#dc.CreateHimaxDataset(config.himax_delay, pickleName, "himax_camera", "bebop/image_raw/compressed", "optitrack/drone", ["optitrack/head", "optitrack/hand"])

	# If you wish to join several .pickle files into one big .pickle, use
	# DatasetCreator.JoinPickleFiles(fileList, newPickleName, folderPath)


if __name__ == '__main__':
    main()