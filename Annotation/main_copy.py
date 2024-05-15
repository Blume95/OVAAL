import numpy as np
from PyQt5.QtCore import *
import sys
import os
import yaml
from argparse import Namespace
from uitls import *
from networks.deeplab import Deeplab


class OVA_Labeling(QMainWindow):
    def __init__(self, args):
        super(OVA_Labeling, self).__init__()
        self.prediction_dir = None
        self.annotated_pixels = None
        self.current_query_dict = None
        self.args = args

        self.label_array = None
        self.current_epoch = 0
        self.hw = None
        self.current_image = None
        self.current_annotated_pkl = None
        self.dict_data_round = None
        self.current_point = None
        self.current_image_path = None
        self.total_points_num = 0
        self.image_info = None
        self.annotation_status = False
        self.current_index_point = 0
        self.image_annotated_list = []
        self.point_label = []
        self.last_x, self.last_y = None, None
        self.manual_selection = False
        self.radius = 4
        self.nth = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Deeplab(class_num=self.args.class_num, output_stride=self.args.stride_total,
                             pretrained=True).to(self.device)
        self.check_selection_round()
        self.query_selector = Query(args)

        self.error_dialog = QErrorMessage()
        self.Widget = QWidget(self)
        self.setWindowTitle("OVA")
        self.setGeometry(1000, 500, 1080, 512)

        # set Toolbar
        self.toolBar = QToolBar("My Toolbar")
        self.toolBar.setEnabled(True)
        self.toolBar.setAutoFillBackground(False)
        self.toolBar.setOrientation(Qt.Horizontal)
        self.toolBar.setIconSize(QSize(60, 40))
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(self.toolBar)

        # set font
        font = QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)

        # label list
        self.label_dock = QDockWidget("Label List", self)
        self.label_dock.setMinimumSize(QSize(200, 1))
        self.label_list_widget = add_list_widget(self.args.used_classes_dict)
        self.label_dock.setWidget(self.label_list_widget)

        # use label to realize a Frames player
        self.image_label = QLabel()
        self.image_label.setBackgroundRole(QPalette.Base)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidgetResizable(True)

        # use label to display current frames and sum frames
        self.current_frame_num = 0
        self.total_frame_num = 0
        self.current_frame_num_label = QLabel(self)
        self.sum_frame_num_label = QLabel(self)

        # add push button
        self.preFrameBtn = QPushButton()
        self.preFrameBtn.setEnabled(False)
        self.preFrameBtn.setIcon(QIcon("../Annotation/icons/previous.png"))
        self.preFrameBtn.clicked.connect(self.preFrame)

        self.nxtFrameBtn = QPushButton()
        self.nxtFrameBtn.setEnabled(False)
        self.nxtFrameBtn.setIcon(QIcon("../Annotation/icons/next.png"))
        self.nxtFrameBtn.clicked.connect(self.nxtFrame)

        # 0. set Actions
        self.actionLoad = make_an_action("Load", "Load Image", '../Annotation/icons/loading.png')
        self.actionLoad.triggered.connect(self.load_images)

        # 1. Selection if human selection ... if auto selection: initial round random / normal round

        # 3. training
        self.actionTraining = make_an_action("Training", "Training model",
                                             "../Annotation/icons/training.png")
        self.actionTraining.triggered.connect(self.training)

        # 4 visualization
        self.actionVis = make_an_action("Visualization", "Show segmentation result",
                                        "../Annotation/icons/vis.png")
        self.actionVis.triggered.connect(self.vis_)
        self.visualization_status = False
        # # 4. Brush
        # self.actionBrush = make_an_action("Brush", "Brush", "../Annotation/icons/brush.svg")
        # # self.actionBrush.triggered.connect(self.enable_human_selection)

        # set progressbar
        self.pbar = QProgressBar(self)

        # Vis action on Toolbar
        self.toolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionTraining)
        self.toolBar.addAction(self.actionVis)
        # self.toolBar.addAction(self.actionBrush)

        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)

        # HBOX LAYOUT
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)
        hboxLayout.addWidget(self.preFrameBtn)
        hboxLayout.addWidget(self.nxtFrameBtn)
        hboxLayout.addWidget(self.current_frame_num_label)
        hboxLayout.addWidget(self.sum_frame_num_label)

        # Vbox Layout
        vboxLayout = QVBoxLayout(self.Widget)
        vboxLayout.addWidget(self.scrollArea)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.pbar)

        self.scrollArea.setWidget(self.image_label)
        self.scrollArea.setVisible(True)

        # enable the next and previous button
        self.preFrameBtn.setEnabled(True)
        self.nxtFrameBtn.setEnabled(True)

        # set the total frames num
        self.sum_frame_num_label.setText("/ " + str(len(self.args.image_names) - 1))
        self.total_frame_num = len(self.args.image_names) - 1

        self.setCentralWidget(self.Widget)

    def vis_(self):
        nth, done = QInputDialog.getInt(self, "Get round num", f"Enter nth [{0}~{self.nth})")
        self.prediction_dir = f"{self.args.dir_root}/{self.args.project_name}/{nth}_query/mask_out"
        if not os.path.exists(self.prediction_dir):
            os.makedirs(self.prediction_dir)
        mask_names = os.listdir(self.prediction_dir)
        if len(mask_names) != len(self.args.image_path_list):
            prediction(self.pbar, self.args, nth, self.device)

        self.visualization_status = True
        self.basic_frame_block()

    def training(self):
        start_training(self.pbar, self.args, self.nth, self.device)
        self.nth += 1
        self.load_images()

    def remove_annotated_pixels(self):
        tem_dict = {}
        for p_img, dict_info in self.current_query_dict.items():
            for k, v in dict_info.items():
                if v != -1:
                    if p_img in tem_dict:
                        tem_dict[p_img][k] = v
                    else:
                        tem_dict[p_img] = {k: v}

        for p_img, dict_info in tem_dict.items():
            for k, v in dict_info.items():
                self.current_query_dict[p_img].pop(k)

        return tem_dict

    def closeEvent(self, event):
        for p_img in self.args.image_path_list:
            self.add_the_removed_pixels(self.annotated_pixels, p_img)

        save_pkl(self.current_query_dict, self.current_annotated_pkl)

        event.accept()

    def add_the_removed_pixels(self, tem_dict, p_img):
        if p_img in tem_dict:
            for k, v in tem_dict[p_img].items():
                self.current_query_dict[p_img][k] = v

    def check_selection_round(self):
        while True:
            self.current_annotated_pkl = f"{self.args.dir_root}/{self.args.project_name}/{self.nth}_query/queries.pkl"
            self.current_weights_list = [
                f"{self.args.dir_root}/{self.args.project_name}/{self.nth}_query/{used_class}_final_epoch_model.pt" for
                used_class in self.args.class_names[1:]]

            if os.path.exists(self.current_annotated_pkl):
                count = 0
                for path in self.current_weights_list:
                    if os.path.exists(path):
                        count += 1
                if count == len(self.current_weights_list):
                    self.nth += 1
                else:
                    break
            else:
                break

    def load_images(self):
        # read annotated points if exists
        self.visualization_status = False
        self.current_annotated_pkl = f"{self.args.dir_root}/{self.args.project_name}/{self.nth}_query/queries.pkl"
        if not os.path.exists(self.current_annotated_pkl):
            # create dir and do selection
            os.makedirs(f"{self.args.dir_root}/{self.args.project_name}/{self.nth}_query/", exist_ok=True)
            self.query_selector(self.nth, self.device, self.pbar, self.model)

        # show points
        self.current_query_dict = read_pkl(self.current_annotated_pkl)
        # remove annotated points
        self.annotated_pixels = self.remove_annotated_pixels()

        self.annotation_status = True
        self.basic_frame_block()
        # load current frame

    def item_selected_changed(self):
        if self.label_list_widget.selectedItems():
            self.point_label.append(self.label_list_widget.selectedItems()[0].text().split(" ")[0])
        else:
            return 0

    def keyPressEvent(self, event):
        # print(event.key())

        def update_index():
            if self.current_index_point == len(self.current_query_dict[self.current_image_path]):
                self.current_index_point = 0
            if self.current_index_point == -1:
                self.current_index_point = len(self.current_query_dict[self.current_image_path]) - 1

        if self.annotation_status:
            if self.item_selected_changed() == 0:
                self.error_dialog.showMessage("select one class in right dock list")
            else:
                self.current_point = list(self.current_query_dict[self.current_image_path].keys())[
                    self.current_index_point]
                if event.key() == 16777220:  # enter
                    point_label = self.point_label[-1]
                    self.current_query_dict[self.current_image_path][self.current_point] = point_label
                    self.image_annotated_list.append(self.current_point)

                    self.point_label = []
                    self.current_index_point += 1
                    update_index()
                    # all pixels have been annotated
                    if -1 not in self.current_query_dict[self.current_image_path].values():
                        self.add_the_removed_pixels(self.annotated_pixels, self.current_image_path)
                        save_pkl(self.current_query_dict, self.current_annotated_pkl)
                        self.image_annotated_list = []
                        self.current_index_point = 0
                        self.nxtFrame()
                elif event.key() == 16777264:  # next
                    self.current_index_point += 1
                    update_index()
                elif event.key() == 16777265:  # prev
                    self.current_index_point -= 1
                    update_index()
                self.show_points(self.current_query_dict)

    def show_points(self, query_dict):
        image = self.current_image.copy()
        if len(list(query_dict[self.current_image_path].keys())) >= 1:
            current_points = list(query_dict[self.current_image_path].keys())[self.current_index_point]
            for k, v in query_dict[self.current_image_path].items():
                image = cv2.circle(image, (k[1], k[0]), self.radius, palette_cs[int(v)], -1)
                if k == current_points:
                    image = cv2.circle(image, (k[1], k[0]), self.radius, (255, 0, 0), -1)

        self.image_label.setPixmap(QPixmap.fromImage(cv_to_qt(image)))

    def preFrame(self):
        if self.current_frame_num == 0:
            self.current_frame_num = self.total_frame_num
        else:
            self.current_frame_num -= 1
        # todo: show human selected points
        self.basic_frame_block()

    def nxtFrame(self):
        if self.current_frame_num == self.total_frame_num:
            self.current_frame_num = 0
        else:
            self.current_frame_num += 1
        # todo: show human selected points
        self.basic_frame_block()

    def basic_frame_block(self):
        # some basic info 1. image path 2. current image 3 hw
        self.current_image_path = self.args.image_path_list[self.current_frame_num]
        self.current_image = cv2.imread(self.current_image_path, -1)[:, :, :3]
        self.hw = self.current_image.shape[:2]

        if self.visualization_status:
            current_label_path = f"{self.prediction_dir}/{self.current_image_path.split('/')[-1]}"
            current_label = cv2.imread(current_label_path, -1)

            self.current_image = 0.4 * self.current_image + 0.6 * current_label
            self.current_image = np.clip(self.current_image, 0, 255).astype(np.uint8)
        current_frame = cv_to_qt(self.current_image)
        self.image_label.setPixmap(QPixmap.fromImage(current_frame))
        self.current_frame_num_label.setText(str(self.current_frame_num))
        if not self.visualization_status:
            self.show_points(self.current_query_dict)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    from args_parameters import Arguments

    args_ = Arguments().parse_args()

    OVA = OVA_Labeling(args_)
    OVA.show()
    sys.exit(app.exec())
