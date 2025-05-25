#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<string>
using namespace cv;
using namespace std;

void myCvtColorToGray(const Mat& src, Mat& dst) {
	CV_Assert(src.type() == CV_8UC3);
	dst.create(src.rows, src.cols, CV_8UC1);
	for (int y = 0; y < src.rows; y++) {
		const Vec3b* srcRow = src.ptr<Vec3b>(y);
		uchar* dstRow = dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++) {
			uchar B = srcRow[x][0];
			uchar G = srcRow[x][1];
			uchar R = srcRow[x][2];
			dstRow[x] = saturate_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
		}
	}
}

class HarrFaceRecongnition
{
public:
	HarrFaceRecongnition(Mat frame)
	{
		myCvtColorToGray(frame, img);
		equalizeHist(img, img);
		result = frame;
		harr_xml = "haarcascade_frontalface_default.xml";
	}

	void GetFace()
	{
		CascadeClassifier object;//检测器对象
		object.load(harr_xml);//加载采样文件
		object.detectMultiScale(img, faces, 1.1, 3);
	}

	void show()
	{
		for (int i = 0; i < faces.size(); i++)
		{
			int x = faces[i].tl().x;
			int y = faces[i].tl().y;
			int width = faces[i].width;
			int height = faces[i].height;
			rectangle(result, Rect(x, y, width, height), Scalar(255, 0, 0));
		}
		imshow("result", result);
	}
private:
	Mat img;
	Mat result;
	string harr_xml;
	vector<Rect> faces;
};

int main()
{
	VideoCapture cap(0);
	Mat frame;
	while (true)
	{
		cap.read(frame);
		if (frame.empty())
		{
			break;
		}
		HarrFaceRecongnition* p = new HarrFaceRecongnition(frame);
		p->GetFace();
		p->show();
		int c = waitKey(10);
		if (c == 27) break;
	}
	cap.release();
	waitKey(0);
	destroyAllWindows();
	return 0;
}