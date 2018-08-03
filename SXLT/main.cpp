#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

#include <inih/INIReader.h>
#include <opencv2/opencv.hpp>

#define _PR(x) std::cout << x << std::endl;


// select your key bindings
#define DESKTOP_KEYS
//#define JETSON_KEYS

#ifdef DESKTOP_KEYS
// pretty much follows http://www.asciitable.com/
constexpr int EXIT_KEY = 27; // escape
constexpr int FINISH_CONTOUR_KEY = 10; // enter (new line)
constexpr int FINISH_IMAGE_KEY = 83; // right arrow
constexpr int RESTART_IMAGE_KEY = 8; // backspace
constexpr int RESTART_CONTOUR_KEY = 47; // forward slash
constexpr int UNDO_LINE_KEY = 122; // z
constexpr int UNDO_POLY_KEY = 90; // shift+z
constexpr int CLASS1 = 49; // 1
constexpr int CLASS2 = 50;
constexpr int CLASS3 = 51;
constexpr int CLASS4 = 52;
constexpr int CLASS5 = 53;
constexpr int CLASS6 = 54;
constexpr int CLASS7 = 55;
constexpr int CLASS8 = 56;
constexpr int CLASS9 = 57; // 9
#endif

#ifdef JETSON_KEYS
constexpr int EXIT_KEY = 27; // escape
constexpr int FINISH_CONTOUR_KEY = 13; // enter (carriage return)
constexpr int FINISH_IMAGE_KEY = 83; // right arrow
constexpr int RESTART_IMAGE_KEY = 8; // backspace
constexpr int RESTART_CONTOUR_KEY = 47; // forward slash
constexpr int UNDO_LINE_KEY = 122; // z
constexpr int UNDO_POLY_KEY = 90; // shift+z
constexpr int CLASS1 = 49; // 1
constexpr int CLASS2 = 50;
constexpr int CLASS3 = 51;
constexpr int CLASS4 = 52;
constexpr int CLASS5 = 53;
constexpr int CLASS6 = 54;
constexpr int CLASS7 = 55;
constexpr int CLASS8 = 56;
constexpr int CLASS9 = 57; // 9
#endif


constexpr int EXIT_FLAG = -1;
constexpr int CONTINUE_FLAG = 0;
constexpr int NEXT_IMG_FLAG = 1;

const int defaultColors[] = {
0, 0, 0, // dummy 0
230, 25, 75, // 1
60, 180, 75, // 2
255, 225, 25, // 3
245, 130, 48, // 4
45, 30, 180, // 5
170, 110, 40, // 6
240, 50, 230, // 7
0, 128, 128, // 8
230, 190, 255, // 9312
128, 0, 0,
0, 0, 128,
128, 128, 128,
};


struct Class {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    std::string name;
};

struct Config {
    std::string outDir;
    float maskOpacity;
};

struct Contour {
    int classIndex;
    std::vector<cv::Point> contour;
};

constexpr char WIN_NAME[] = "XSLT";

cv::Mat base_image;
cv::Mat label_mask;
cv::Mat display;
std::vector<Contour> contours;
Config config;
std::vector<Class> classes(10);
Contour currentContour;
int currentClassIndex = 1;


using namespace std;
using namespace cv;

static string getFileName(const string& filePath, bool withExtension = true, char seperator = '/')
{
    // Get last dot position
    std::size_t dotPos = filePath.rfind('.');
    std::size_t sepPos = filePath.rfind(seperator);
    size_t firstPos;
    if (sepPos == string::npos) {
        firstPos = 0;
    } else {
        firstPos = sepPos + 1;
    }

    size_t lastPos;
    if (dotPos == string::npos) {
        lastPos = string::npos;
    } else {
        lastPos = withExtension ? string::npos : dotPos;
    }

    return filePath.substr(firstPos, lastPos - firstPos);
}

inline static bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

inline static string labelPathFromImage(const string& imagePath) {
    return config.outDir + getFileName(imagePath, false) + ".png";
}

static vector<vector<Point>> makeInput(Contour c) {
    // copies but who really cares
    vector<vector<Point>> a;
    a.push_back(c.contour);
    return a;
}

static void rerenderAll(bool renderInProgressContour) {
//    _PR("start render")
    label_mask = Mat::zeros(base_image.size(), base_image.type());
//    _PR("askljd")
    for (Contour c : contours) {
        Class cl = classes.at(c.classIndex);
        fillPoly(label_mask, makeInput(c), Scalar(cl.b, cl.g, cl.r));
    }
    // render lines of current poly
//    _PR("flag")
    if (renderInProgressContour) {
        for (int i = 0; i + 1 < currentContour.contour.size(); i++) {
            Class cl = classes.at(currentContour.classIndex);
            line(label_mask, currentContour.contour.at(i), currentContour.contour.at(i+1), Scalar(cl.b, cl.g, cl.r));
        }
    }

    addWeighted(base_image, 1-config.maskOpacity, label_mask, config.maskOpacity, 0.0, display);
    imshow(WIN_NAME, display);
}

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != EVENT_LBUTTONDOWN )
        return;

    currentContour.contour.emplace_back(Point(x,y));
    rerenderAll(true);
}

static void changeClass(int classIndex) {
    currentClassIndex = classIndex;
    currentContour.classIndex = classIndex;
}

static void updateWindowTitle(string imageName) {
    setWindowTitle(WIN_NAME, "XLT | " + imageName + " | " + classes.at(currentClassIndex).name);
}

static int handleKeyPress(int code) {
    cout << "Handling keypress with code " << code << endl;
    switch (code) {
    case EXIT_KEY:
        return EXIT_FLAG;
    case FINISH_CONTOUR_KEY:
//        _PR("fin cont")
        contours.push_back(currentContour);
        currentContour = Contour();
        currentContour.contour = vector<Point>();
        currentContour.classIndex = currentClassIndex;
        break;
    case FINISH_IMAGE_KEY:
        return NEXT_IMG_FLAG;
    case RESTART_IMAGE_KEY:
        contours = vector<Contour>();
        // fallthrough
    case RESTART_CONTOUR_KEY:
        currentContour = Contour();
        currentContour.contour = vector<Point>();
        currentContour.classIndex = currentClassIndex;
        break;
    case UNDO_LINE_KEY:
        if (!currentContour.contour.empty())
            currentContour.contour.pop_back();
        break;
    case UNDO_POLY_KEY:
        if (!contours.empty())
            contours.pop_back();
        break;
    case CLASS1:
        changeClass(1);
        break;
    case CLASS2:
        changeClass(2);
        break;
    case CLASS3:
        changeClass(3);
        break;
    case CLASS4:
        changeClass(4);
        break;
    case CLASS5:
        changeClass(5);
        break;
    case CLASS6:
        changeClass(6);
        break;
    case CLASS7:
        changeClass(7);
        break;
    case CLASS8:
        changeClass(8);
        break;
    case CLASS9:
        changeClass(9);
        break;
    default:
        break;
    }
    return CONTINUE_FLAG;
}

void saveLabel(string imageName) {
    rerenderAll(false);
    string out = labelPathFromImage(imageName);
    cout << "Saving image to " << out << endl;
    imwrite(out, label_mask);
}

int main(int argc, char *argv[]) {
    INIReader reader("SXLT.ini");

    if (reader.ParseError() < 0) {
        std::cout << "Can't load 'SXLT.ini'\n";
        return 1;
    }

    config.outDir = reader.Get("", "output_dir", "img");
    config.maskOpacity = static_cast<float>(reader.GetReal("", "mask_opacity", 0.3));

    for (int i = 1; i <= 9; i++) {
        string className("class");
        className += to_string(i);
        Class c;
        c.r = static_cast<unsigned char>(reader.GetInteger(className, "red", defaultColors[i * 3] + 0));
        c.g = static_cast<unsigned char>(reader.GetInteger(className, "green", defaultColors[i * 3 + 1]));
        c.b = static_cast<unsigned char>(reader.GetInteger(className, "blue", defaultColors[i * 3 + 2]));
        c.name = reader.Get(className, "name", className);
        classes.at(i) = c;
    }

    cout << "Loading classes from config..." << endl;
    for (Class c : classes) {
        _PR("class " << c.name)
        _PR("r " << static_cast<unsigned>(c.r))
        _PR("g " << static_cast<unsigned>(c.g))
        _PR("b " << static_cast<unsigned>(c.b))
    }
    _PR("Using label directory " << config.outDir)
    _PR("Using mask opacity " << config.maskOpacity)
    cout << "Begin Labeling" << endl;

    namedWindow(WIN_NAME, WINDOW_NORMAL);
    moveWindow(WIN_NAME, 20, 20);
    resizeWindow(WIN_NAME, Size(800, 800));
    setMouseCallback(WIN_NAME, onMouse, nullptr);

    if (argc <= 1) {
        cout << "Must Specify one or more images" << endl;
        return 1;
    }

    for (int idx = 1; idx < argc; idx++) {
        string imageName(argv[idx]);

        if (file_exists(labelPathFromImage(imageName))) {
            cout << "Found label for " << imageName << ", continuing..." << endl;
            continue;
        }

        base_image = imread(imageName);
        handleKeyPress(RESTART_IMAGE_KEY); // reset all state

        bool wait = true;
        while (wait) {
            rerenderAll(true);
            updateWindowTitle(imageName);
            switch (handleKeyPress(waitKey(0))) {
                case EXIT_FLAG:
                    return 0;
                case CONTINUE_FLAG:
                    break;
                case NEXT_IMG_FLAG:
                    wait = false;
                    break;
                default:
                    break;
            };
        }
        // save image
        saveLabel(imageName);
    }
    return 0;
}