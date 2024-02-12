'''
UnTop2d: The Graphical User Interface
'''


import sys, os, shutil, time, glob, copy, threading, vtk
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.optimize import brentq
import pyvista as pv
from pyvistaqt import QtInteractor
from unstructured_topopt import Structure


# Config parameters
FONT_FAMILY = "Arial"

VISTA_SHOW_LABEL = False
VISTA_SHOW_GRID = True
VISTA_SHOW_COLORBAR = True
VISTA_LABEL_FONTSIZE = 16
VISTA_SCREENSHOT_FORMAT = 'png'
VISTA_SCREENSHOT_SIZE = (2400, 1600)

MATPLOT_FIGSIZE=(6.4, 4.8)
MATPLOT_EXPORT_FORMAT = 'svg'
MATPLOT_AXES_LABEL_FONTSIZE = 16
MATPLOT_TICKS_LABEL_FONTSIZE = 12
MATPLOT_AXES_LABEL_COLOR = 'black' #'#525252'

DPI = 500  # For image export; pyvista, matplotlib

# Set recursion limit; default 1000
sys.setrecursionlimit(10000000)

pv.set_plot_theme("document")  # Options: ParaView, default
plt.rcParams["font.family"] = FONT_FAMILY


class OptimizeThread(QtCore.QThread):

    def __init__(self, struct):
        QtCore.QThread.__init__(self)

        self.struct = struct

    def run(self):
        self.struct.optimize()
        #self.struct.write_mesh(output_filename='output_files/mesh.vtk', write_format='vtk-ascii')


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('UnTop2D')

        # Menubar
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')
        action_menu = main_menu.addMenu('&Action')
        set_menu = main_menu.addMenu('&Set')
        analysis_menu = main_menu.addMenu('&Analysis')
        postprocess_menu = main_menu.addMenu('&Postprocess')

        # Create project
        self.create_proj_action = QtWidgets.QAction('&Create Project from Input File')
        self.create_proj_action.triggered.connect(self.create_project_)
        file_menu.addAction(self.create_proj_action)

        # Open project
        self.open_proj_action = QtWidgets.QAction('&Open Project Directory')
        self.open_proj_action.triggered.connect(self.open_project)
        file_menu.addAction(self.open_proj_action)

        # Close project
        self.close_proj_action = QtWidgets.QAction('&Close Project')
        self.close_proj_action.triggered.connect(self.close_project)
        file_menu.addAction(self.close_proj_action)

        self.screenshot_action = QtWidgets.QAction('&Take Screenshot')
        self.screenshot_action.triggered.connect(self.take_screenshot)
        file_menu.addAction(self.screenshot_action)

        # Stop opt
        self.stop_opt_action = QtWidgets.QAction('&Stop Optimization')
        self.stop_opt_action.triggered.connect(self.stop_optimization)
        action_menu.addAction(self.stop_opt_action)

        # Set optimization parameters
        self.set_opt_params_action = QtWidgets.QAction('&Optimization Params')
        self.set_opt_params_action.triggered.connect(self.manage_opt_params)
        set_menu.addAction(self.set_opt_params_action)

        # Set loads
        self.set_loads_action = QtWidgets.QAction('&Loads')
        self.set_loads_action.triggered.connect(self.manage_loads)
        set_menu.addAction(self.set_loads_action)

        # Set BCs
        self.set_bcs_action = QtWidgets.QAction('&BCs')
        self.set_bcs_action.triggered.connect(self.manage_bcs)
        set_menu.addAction(self.set_bcs_action)

        # Set freeze elements
        self.set_freeze_action = QtWidgets.QAction('&Freeze Elements')
        self.set_freeze_action.triggered.connect(self.manage_freeze)
        set_menu.addAction(self.set_freeze_action)

        # Check mesh
        self.check_mesh_action = QtWidgets.QAction('&Check Mesh')
        self.check_mesh_action.triggered.connect(self.check_mesh)
        analysis_menu.addAction(self.check_mesh_action)

        # Plot compliance
        self.plot_compl_action = QtWidgets.QAction('&Plot Compliance')
        self.plot_compl_action.triggered.connect(self.plot_compl)
        analysis_menu.addAction(self.plot_compl_action)

        # Plot max. relative density change
        self.plot_max_xchange_action = QtWidgets.QAction('&Plot Max. Rel. M. Density Change')
        self.plot_max_xchange_action.triggered.connect(self.plot_max_xchange)
        analysis_menu.addAction(self.plot_max_xchange_action)

        # Filter elements
        self.filter_elem_action = QtWidgets.QAction('&Filter Elements')
        self.filter_elem_action.triggered.connect(self.filter_elements)
        postprocess_menu.addAction(self.filter_elem_action)

        main_widget = QtWidgets.QWidget()
        self.main_vlay = QtWidgets.QVBoxLayout()
        #self.main_vlay.setAlignment(QtCore.Qt.AlignTop)
        main_widget.setLayout(self.main_vlay)
        self.setCentralWidget(main_widget)

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setFont(QtGui.QFont(FONT_FAMILY, 12))
        self.statusBar.setStyleSheet("QStatusBar{color:#990000;font-weight:bold;}")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Please Create or Open a Project')

        self.file_index = 0
        self.draw_widget = None

        self.set_submenu_btn_enabled(False)

    def set_submenu_btn_enabled(self, state=True):
        self.close_proj_action.setEnabled(state)
        self.screenshot_action.setEnabled(state)
        self.set_opt_params_action.setEnabled(state)
        self.set_loads_action.setEnabled(state)
        self.set_bcs_action.setEnabled(state)
        self.set_freeze_action.setEnabled(state)
        self.stop_opt_action.setEnabled(state)
        self.check_mesh_action.setEnabled(state)
        self.filter_elem_action.setEnabled(state)
        self.plot_compl_action.setEnabled(state)
        self.plot_max_xchange_action.setEnabled(state)

    def open_project(self):
        directory_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder"))

        if directory_name:
            self.delete_items_layout(self.main_vlay)

            input_filename = directory_name + '/input_file.txt'
            os.chdir(directory_name)

            self.statusBar.showMessage('Importing Project ...')
            self.init_topology_structure(input_filename=input_filename)
            self.add_buttons()
            self.init_mesh()

            self.set_submenu_btn_enabled(True)

    def close_project(self):
        self.delete_items_layout(self.main_vlay)

        self.set_submenu_btn_enabled(False)

        self.statusBar.showMessage('Please Create or Open a Project')

    def take_screenshot(self):
        self.draw_widget.take_screenshot()
        self.statusBar.showMessage('Screenshot taken. Please check the project directory.')

    def delete_items_layout(self, layout):  # Taken from https://riverbankcomputing.com/pipermail/pyqt/2009-November/025214.html
     if layout is not None:
         while layout.count():
             item = layout.takeAt(0)
             widget = item.widget()
             if widget is not None:
                 widget.setParent(None)
             else:
                 self.delete_items_layout(item.layout())

    def create_project_(self):
        self.create_proj_dlg = CreateProjWindow()
        self.create_proj_dlg.create_btn.clicked.connect(self.create_project)

    def create_project(self):
        self.delete_items_layout(self.main_vlay)

        pname = self.create_proj_dlg.pname_lineedit.text()
        ploc = self.create_proj_dlg.ploc_lineedit.text()
        inputloc = self.create_proj_dlg.inputloc_lineedit.text()

        proj_dir = ploc + '/' + pname
        proj_input_file = proj_dir + '/input_file.txt'

        os.mkdir(proj_dir)
        os.chdir(proj_dir)
        shutil.copyfile(inputloc, proj_input_file)

        self.statusBar.showMessage('Creating Project ...')
        self.init_topology_structure(input_filename=proj_input_file)
        self.add_buttons()
        self.init_mesh()
        #self.volfrac_lineedit.setText(str(self.struct.volfrac))

        self.set_submenu_btn_enabled(True)
        self.create_proj_dlg.close()

    def reload_from_inputfile(self):
        self.delete_items_layout(self.main_vlay)

        self.statusBar.showMessage('Reloading Project ...')
        self.init_topology_structure(input_filename='input_file.txt')
        self.add_buttons()
        self.init_mesh()

    def manage_opt_params(self):
        self.opt_params_dlg = QtWidgets.QDialog(None, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.opt_params_dlg.setWindowTitle("Set Optimization Parameters")

        vlay = QtWidgets.QVBoxLayout()

        grid_lay = QtWidgets.QGridLayout()

        volfrac_label = QtWidgets.QLabel("Volume fraction:")
        self.volfrac_dspinbox = QtWidgets.QDoubleSpinBox()
        self.volfrac_dspinbox.setMinimum(0)
        self.volfrac_dspinbox.setMaximum(1)
        self.volfrac_dspinbox.setDecimals(2)
        grid_lay.addWidget(volfrac_label, 0, 0)
        grid_lay.addWidget(self.volfrac_dspinbox, 0, 1)

        penal_label = QtWidgets.QLabel("Penalization power:")
        self.penal_spinbox = QtWidgets.QSpinBox()
        grid_lay.addWidget(penal_label, 1, 0)
        grid_lay.addWidget(self.penal_spinbox, 1, 1)

        max_x_change_thresh_label = QtWidgets.QLabel("Max. Rel. M. Density Change Thresh.:")
        self.max_x_change_thresh_dspinbox = QtWidgets.QDoubleSpinBox()
        self.max_x_change_thresh_dspinbox.setDecimals(4)
        self.max_x_change_thresh_dspinbox.setMinimum(0)
        self.max_x_change_thresh_dspinbox.setMaximum(1)
        grid_lay.addWidget(max_x_change_thresh_label, 2, 0)
        grid_lay.addWidget(self.max_x_change_thresh_dspinbox, 2, 1)

        max_iter_label = QtWidgets.QLabel("Max. Iteration:")
        self.max_iter_spinbox = QtWidgets.QSpinBox()
        self.max_iter_spinbox.setMinimum(0)
        self.max_iter_spinbox.setMaximum(2000)
        grid_lay.addWidget(max_iter_label, 3, 0)
        grid_lay.addWidget(self.max_iter_spinbox, 3, 1)

        self.set_opt_params()

        save_hlay = QtWidgets.QHBoxLayout()

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.write_opt_params)

        save_hlay.addStretch()
        save_hlay.addWidget(save_btn)

        vlay.addLayout(grid_lay)
        vlay.addLayout(save_hlay)

        self.opt_params_dlg.setLayout(vlay)
        self.opt_params_dlg.show()

    def set_opt_params(self):
        self.volfrac_dspinbox.setValue(self.struct.volfrac)
        self.penal_spinbox.setValue(self.struct.penal)
        self.max_x_change_thresh_dspinbox.setValue(self.struct.max_x_change_thresh)
        self.max_iter_spinbox.setValue(self.struct.max_iter)

    def get_opt_params(self):
        volfrac = self.volfrac_dspinbox.value()
        penal = self.penal_spinbox.value()
        max_x_change_thresh = self.max_x_change_thresh_dspinbox.value()
        max_iter = self.max_iter_spinbox.value()

        return [volfrac, penal, max_x_change_thresh, max_iter]

    def write_opt_params(self):
        volfrac, penal, max_x_change_thresh, max_iter = self.get_opt_params()

        f = open('input_file.txt', 'r')
        lines = f.readlines()
        f.close()

        f = open('input_file.txt', 'w')
        found_max_x_change_thresh = False  # to determine if the field exist in input file and to create if not found
        found_max_iter = False
        for line in lines:
            _line = line.strip()

            if _line.startswith('VOLFRAC'):
                line = "VOLFRAC {}\n".format(volfrac)
            elif _line.startswith('PENAL'):
                line = "PENAL {}\n".format(penal)
            elif _line.startswith('MAX_X_CHANGE_THRESH'):
                line = "MAX_X_CHANGE_THRESH {}\n".format(max_x_change_thresh)
                found_max_x_change_thresh = True
            elif _line.startswith('MAX_ITER'):
                line = "MAX_ITER {}\n".format(max_iter)
                found_max_iter = True

            f.write(line)

        if not found_max_x_change_thresh:
            f.write('\nMAX_X_CHANGE_THRESH {}\n'.format(max_x_change_thresh))
        if not found_max_iter:
            f.write('\nMAX_ITER {}\n'.format(max_iter))

        f.close()

        self.opt_params_dlg.close()
        self.reload_from_inputfile()

    def manage_loads(self):
        self.loads_dlg = QtWidgets.QDialog(None, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.loads_dlg.setWindowTitle("Set Loads")

        vlay = QtWidgets.QVBoxLayout()

        help_txt = QtWidgets.QLabel("Put negative value in force field for negative direction along axis.")
        self.loads_table = QtWidgets.QTableWidget()
        self.set_loads()
        self.loads_table.setHorizontalHeaderLabels(['Node Number', 'Fx', 'Fy'])

        btn_hlay = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedWidth(25)
        add_btn.clicked.connect(self.add_load)
        delete_btn = QtWidgets.QPushButton("-")
        delete_btn.setFixedWidth(25)
        delete_btn.clicked.connect(self.delete_load)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.write_loads)

        btn_hlay.addWidget(delete_btn)
        btn_hlay.addWidget(add_btn)
        btn_hlay.addStretch()
        btn_hlay.addWidget(save_btn)

        vlay.addWidget(help_txt)
        vlay.addWidget(self.loads_table)
        vlay.addLayout(btn_hlay)

        self.loads_dlg.setLayout(vlay)
        self.loads_dlg.show()

    def add_load(self):
        self.loads_table.insertRow(self.loads_table.rowCount())

    def delete_load(self):
        row_count = self.loads_table.rowCount()
        if row_count > 0:
            self.loads_table.removeRow(row_count - 1)

    def set_loads(self):
        new_load_dict = {}
        for dof in self.struct.load_dict.keys():
            if dof % 2 == 0:
                node_num = int(dof / 2)
                try:
                    new_load_dict[node_num]['fy'] = self.struct.load_dict[dof]
                except:
                    new_load_dict[node_num] = {'fx': 0, 'fy': self.struct.load_dict[dof]}
            else:
                node_num = int((dof + 1) / 2)
                try:
                    new_load_dict[node_num]['fx'] = self.struct.load_dict[dof]
                except:
                    new_load_dict[node_num] = {'fx': self.struct.load_dict[dof], 'fy': 0}

        self.loads_table.setColumnCount(3)
        self.loads_table.setRowCount(len(new_load_dict))

        for i, node_num in enumerate(new_load_dict.keys()):
            self.loads_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(node_num)))
            self.loads_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(new_load_dict[node_num]['fx'])))
            self.loads_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(new_load_dict[node_num]['fy'])))

    def get_loads(self):
        row_count = self.loads_table.rowCount()
        new_load_dict = {}
        for i in range(row_count):
            node_num_str = self.loads_table.item(i, 0).text().strip() if self.loads_table.item(i, 0) else None
            node_num = int(node_num_str) if node_num_str else None
            if node_num:
                fx_str = self.loads_table.item(i, 1).text().strip() if self.loads_table.item(i, 1) else ""
                fx = float(fx_str) if fx_str else 0
                fy_str = self.loads_table.item(i, 2).text().strip() if self.loads_table.item(i, 2) else ""
                fy = float(fy_str) if fy_str else 0

                if fx:
                    dof = int(2 * node_num - 1)
                    new_load_dict[dof] = fx
                if fy:
                    dof = int(2 * node_num)
                    new_load_dict[dof] = fy
            else:
                continue

        return new_load_dict

    def write_loads(self):
        new_load_dict = self.get_loads()

        f = open('input_file.txt', 'r')
        lines = f.readlines()
        f.close()

        f = open('input_file.txt', 'w')
        # Delete previous loads from input file
        in_load = False
        for line in lines:
            _line = line.strip()
            if _line.startswith('LOAD') and _line.endswith('START'):
                in_load = True
                continue
            if _line.startswith('LOAD') and _line.endswith('END'):
                in_load = False
                continue
            if in_load:
                continue

            f.write(line)
        # Now write updated loads
        f.write('\nLOAD START\n')
        for dof in new_load_dict.keys():
            f.write('{:.0f}    {}\n'.format(dof, new_load_dict[dof]))
        f.write('LOAD END\n')
        f.close()

        self.loads_dlg.close()
        self.reload_from_inputfile()

    def manage_bcs(self):
        self.bcs_dlg = QtWidgets.QDialog(None, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.bcs_dlg.setWindowTitle("Set BCs")

        vlay = QtWidgets.QVBoxLayout()

        self.bcs_table = QtWidgets.QTableWidget()
        self.set_bcs()
        self.bcs_table.setHorizontalHeaderLabels(['Node Number', 'X-constraint', 'Y-constraint'])

        btn_hlay = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedWidth(25)
        add_btn.clicked.connect(self.add_bc)
        delete_btn = QtWidgets.QPushButton("-")
        delete_btn.setFixedWidth(25)
        delete_btn.clicked.connect(self.delete_bc)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.write_bcs)

        btn_hlay.addWidget(delete_btn)
        btn_hlay.addWidget(add_btn)
        btn_hlay.addStretch()
        btn_hlay.addWidget(save_btn)

        vlay.addWidget(self.bcs_table)
        vlay.addLayout(btn_hlay)

        self.bcs_dlg.setLayout(vlay)
        self.bcs_dlg.show()

    def add_bc(self):
        self.bcs_table.insertRow(self.bcs_table.rowCount())

        i = self.bcs_table.rowCount() - 1
        self.bcs_table.setCellWidget(i, 1, QtWidgets.QCheckBox())
        self.bcs_table.setCellWidget(i, 2, QtWidgets.QCheckBox())

    def delete_bc(self):
        row_count = self.bcs_table.rowCount()
        if row_count > 0:
            self.bcs_table.removeRow(row_count - 1)

    def set_bcs(self):
        new_bc_dict = {}
        for dof in self.struct.bc_dict.keys():
            if dof % 2 == 0:
                node_num = int(dof / 2)
                try:
                    new_bc_dict[node_num]['y_constr'] = True
                except:
                    new_bc_dict[node_num] = {'x_constr': False, 'y_constr': True}
            else:
                node_num = int((dof + 1) / 2)
                try:
                    new_bc_dict[node_num]['x_constr'] = True
                except:
                    new_bc_dict[node_num] = {'x_constr': True, 'y_constr': False}

        self.bcs_table.setColumnCount(3)
        self.bcs_table.setRowCount(len(new_bc_dict))

        for i, node_num in enumerate(new_bc_dict.keys()):
            self.bcs_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(node_num)))

            x_const_checkbox = QtWidgets.QCheckBox()
            x_const_checkbox.setChecked(new_bc_dict[node_num]['x_constr'])
            self.bcs_table.setCellWidget(i, 1, x_const_checkbox)

            y_const_checkbox = QtWidgets.QCheckBox()
            y_const_checkbox.setChecked(new_bc_dict[node_num]['y_constr'])
            self.bcs_table.setCellWidget(i, 2, y_const_checkbox)

    def get_bcs(self):
        row_count = self.bcs_table.rowCount()
        new_bc_dict = {}
        for i in range(row_count):
            node_num_str = self.bcs_table.item(i, 0).text().strip() if self.bcs_table.item(i, 0) else None
            node_num = int(node_num_str) if node_num_str else None
            if node_num:
                x_constr = self.bcs_table.cellWidget(i, 1).isChecked()
                y_constr = self.bcs_table.cellWidget(i, 2).isChecked()

                if x_constr:
                    dof = int(2 * node_num - 1)
                    new_bc_dict[dof] = 0
                if y_constr:
                    dof = int(2 * node_num)
                    new_bc_dict[dof] = 0
            else:
                continue

        return new_bc_dict

    def write_bcs(self):
        new_bc_dict = self.get_bcs()

        f = open('input_file.txt', 'r')
        lines = f.readlines()
        f.close()

        f = open('input_file.txt', 'w')
        # Delete previous BCs from input file
        in_bc = False
        for line in lines:
            _line = line.strip()
            if _line.startswith('BC') and _line.endswith('START'):
                in_bc = True
                continue
            if _line.startswith('BC') and _line.endswith('END'):
                in_bc = False
                continue
            if in_bc:
                continue

            f.write(line)
        # Now write updated loads
        f.write('\nBC START\n')
        for dof in new_bc_dict.keys():
            f.write('{:.0f}    {}\n'.format(dof, new_bc_dict[dof]))
        f.write('BC END\n')
        f.close()

        self.bcs_dlg.close()
        self.reload_from_inputfile()

    def manage_freeze(self):
        self.freeze_dlg = QtWidgets.QDialog(None, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint)
        self.freeze_dlg.setWindowTitle("Set Freeze Elements")

        vlay = QtWidgets.QVBoxLayout()

        self.freeze_walls_checkbox = QtWidgets.QCheckBox("Freeze wall elements?")

        self.freeze_table = QtWidgets.QTableWidget()
        self.set_freeze()
        self.freeze_table.setHorizontalHeaderLabels(['Element Number'])

        btn_hlay = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("+")
        add_btn.setFixedWidth(25)
        add_btn.clicked.connect(self.add_freeze_element)
        delete_btn = QtWidgets.QPushButton("-")
        delete_btn.setFixedWidth(25)
        delete_btn.clicked.connect(self.delete_freeze_element)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.write_freeze)

        btn_hlay.addWidget(delete_btn)
        btn_hlay.addWidget(add_btn)
        btn_hlay.addStretch()
        btn_hlay.addWidget(save_btn)

        vlay.addWidget(self.freeze_walls_checkbox)
        vlay.addWidget(self.freeze_table)
        vlay.addLayout(btn_hlay)

        self.freeze_dlg.setLayout(vlay)
        self.freeze_dlg.show()

    def add_freeze_element(self):
        self.freeze_table.insertRow(self.freeze_table.rowCount())

    def delete_freeze_element(self):
        row_count = self.freeze_table.rowCount()
        if row_count > 0:
            self.freeze_table.removeRow(row_count - 1)

    def set_freeze(self):
        self.freeze_table.setColumnCount(1)
        self.freeze_table.setRowCount(len(self.struct.freeze_elem_lst))

        for i, elem_num in enumerate(self.struct.freeze_elem_lst):
            self.freeze_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(elem_num)))

        self.freeze_walls_checkbox.setChecked(self.struct.freeze_walls)

    def get_freeze(self):
        row_count = self.freeze_table.rowCount()
        new_freeze_elem_lst = []
        for i in range(row_count):
            elem_num_str = self.freeze_table.item(i, 0).text().strip() if self.freeze_table.item(i, 0) else None
            elem_num = int(elem_num_str) if elem_num_str else None
            if elem_num:
                new_freeze_elem_lst.append(elem_num)
            else:
                continue

        return new_freeze_elem_lst, self.freeze_walls_checkbox.isChecked()

    def write_freeze(self):
        new_freeze_elem_lst, freeze_walls = self.get_freeze()

        f = open('input_file.txt', 'r')
        lines = f.readlines()
        f.close()

        f = open('input_file.txt', 'w')
        # Delete previous freeze element numbers from input file
        in_freeze = False
        for line in lines:
            _line = line.strip()
            if _line.startswith('FREEZE') and _line.endswith('START'):
                in_freeze = True
                continue
            if _line.startswith('FREEZE') and _line.endswith('END'):
                in_freeze = False
                continue
            if in_freeze:
                continue

            if _line.startswith('FREEZE_WALLS'):
                line = "FREEZE_WALLS {}\n".format("TRUE" if freeze_walls else "FALSE")

            f.write(line)

        # Now write updated loads
        f.write('\nFREEZE START\n')
        for elem_num in new_freeze_elem_lst:
            f.write('{:.0f}\n'.format(elem_num))
        f.write('FREEZE END\n')
        f.close()

        self.freeze_dlg.close()
        self.reload_from_inputfile()

    def check_mesh(self):
        self.check_mesh_txtedit = QtWidgets.QTextEdit()
        self.check_mesh_txtedit.setText(self.struct.check_mesh())
        self.check_mesh_txtedit.setFont(QtGui.QFont(FONT_FAMILY, 12))

        self.check_mesh_txtedit.setWindowTitle('Check Mesh')
        self.check_mesh_txtedit.show()

    def filter_elements(self):
        self.filter_dlg = FilterElements(self.struct, self.file_index, self.statusBar)

        self.filter_dlg.show()

    def plot_compl(self):
        self.plot_compl_dlg = PlotCompliance()
        self.plot_compl_dlg.show()

    def plot_max_xchange(self):
        self.plot_max_xchange_dlg = PlotMaxXChange()
        self.plot_max_xchange_dlg.show()

    def init_topology_structure(self, input_filename='input_file.txt'):
        self.struct = Structure()
        self.struct.parse_input_file(filename=input_filename)
        self.struct.create_object_tables()
        self.struct.prepare_bc_load()
        self.struct.find_adjacent_elem()  # According to radius

    def init_mesh(self):
        self.bounding_box = self.struct.get_bounding_box()

        X = self.struct.get_X_array()
        #self.draw_widget = DrawMesh(self.struct.element_table, X, self.bounding_box)
        self.draw_widget = VistaMesh(self.struct.node_table, self.struct.element_table, X)

        if not os.path.isdir('X'):
            os.mkdir('X')
        X.tofile('X/0.dat')

        self.main_vlay.addWidget(self.draw_widget)

        self.statusBar.showMessage('Iteration: 0')

    def add_buttons(self):
        opt_hlay = QtWidgets.QHBoxLayout()

        optimize_btn = QtWidgets.QPushButton('Optimize')
        optimize_btn.setMinimumHeight(40)
        optimize_btn.setStyleSheet("background-color: #73C6B6")
        optimize_btn.clicked.connect(self.optimize)

        first_btn = QtWidgets.QPushButton('First')
        first_btn.setStyleSheet("background-color: #D5D8DC")
        first_btn.clicked.connect(self.draw_first)

        last_btn = QtWidgets.QPushButton('Last')
        last_btn.setStyleSheet("background-color: #D5D8DC")
        last_btn.clicked.connect(self.draw_last)

        next_btn = QtWidgets.QPushButton('>')
        next_btn.setStyleSheet("background-color: #D5D8DC")
        next_btn.clicked.connect(self.draw_next)

        prev_btn = QtWidgets.QPushButton('<')
        prev_btn.setStyleSheet("background-color: #D5D8DC")
        prev_btn.clicked.connect(self.draw_prev)

        opt_hlay.addWidget(optimize_btn)
        opt_hlay.addStretch()
        opt_hlay.addWidget(first_btn)
        opt_hlay.addWidget(prev_btn)
        opt_hlay.addWidget(next_btn)
        opt_hlay.addWidget(last_btn)

        opt_group = QtWidgets.QGroupBox()
        opt_group.setLayout(opt_hlay)

        control_hlay = QtWidgets.QHBoxLayout()
        control_hlay.setContentsMargins(10, 0, 10, 0)
        control_hlay.addWidget(opt_group)

        self.main_vlay.addLayout(control_hlay)

    def optimize(self):
        # Remove previous optimization data
        fileloc_lst = glob.glob('X/*')
        for fileloc in fileloc_lst:
            os.remove(fileloc)
        self.struct.get_X_array().tofile('X/0.dat')
        self.draw_first()

        opt = threading.Thread(target=self.struct.optimize)
        opt.start()

    def stop_optimization(self):
        self.struct.stop_optimization()

    def draw_first(self):
        file_lst = self.get_files()

        if len(file_lst) > 0:
            self.file_index = 0

            X = np.fromfile('X/'+file_lst[0], dtype=np.float64)

            self.draw_widget.plotter.update_scalars(X)

            self.main_vlay.addWidget(self.draw_widget)
            self.statusBar.showMessage('Iteration: 0')

    def draw_last(self):
        file_lst = self.get_files()

        if len(file_lst) > 0:
            self.file_index = len(file_lst)-1

            X = np.fromfile('X/'+file_lst[-1], dtype=np.float64)

            self.draw_widget.plotter.update_scalars(X)

            self.main_vlay.addWidget(self.draw_widget)
            self.statusBar.showMessage('Iteration: '+str(self.file_index))

    def draw_next(self):
        file_lst = self.get_files()

        if len(file_lst)-1 > self.file_index:
            self.file_index += 1

            X = np.fromfile('X/'+file_lst[self.file_index], dtype=np.float64)

            self.draw_widget.plotter.update_scalars(X)

            self.main_vlay.addWidget(self.draw_widget)
            self.statusBar.showMessage('Iteration: '+str(self.file_index))

    def draw_prev(self):
        file_lst = self.get_files()

        if len(file_lst)>0 and  self.file_index>0:
            self.file_index -= 1

            X = np.fromfile('X/'+file_lst[self.file_index], dtype=np.float64)

            self.draw_widget.plotter.update_scalars(X)

            self.main_vlay.addWidget(self.draw_widget)
            self.statusBar.showMessage('Iteration: '+str(self.file_index))

    @staticmethod
    def get_files():
        file_lst_unf = []
        for (_, __, files) in os.walk('X'):
            file_lst_unf.extend(files)
            break

        # Filter files
        file_lst_int = []
        for file in file_lst_unf:
            if file.endswith('.dat'):
                file_lst_int.append(int(file.replace('.dat', '')))
        file_lst_int.sort()

        file_lst = [str(i)+'.dat' for i in file_lst_int]

        return file_lst


class VistaMesh(QtWidgets.QFrame):

    def __init__(self, node_table, element_table, X, show_grid=VISTA_SHOW_GRID, show_label=VISTA_SHOW_LABEL):
        super().__init__()

        # PyVista interactor
        main_vlay = QtWidgets.QVBoxLayout()
        self.plotter = QtInteractor(self)
        main_vlay.addWidget(self.plotter.interactor)
        self.init_buttons(main_vlay)
        self.setLayout(main_vlay)

        self.create_unstructured_grid(node_table, element_table, X, show_grid, show_label)

    def init_buttons(self, main_vlay):
        btn_hlay = QtWidgets.QHBoxLayout()

        xy_view_btn = QtWidgets.QPushButton('XY')
        xy_view_btn.setStyleSheet("background-color: #D5D8DC")
        xy_view_btn.setAutoDefault(False)
        xy_view_btn.clicked.connect(self.xy_view)
        isometric_view_btn = QtWidgets.QPushButton('Isometric')
        isometric_view_btn.setStyleSheet("background-color: #D5D8DC")
        isometric_view_btn.setAutoDefault(False)
        isometric_view_btn.clicked.connect(self.isometric_view)
        btn_hlay.addStretch(1)
        btn_hlay.addWidget(xy_view_btn)
        btn_hlay.addWidget(isometric_view_btn)

        main_vlay.addLayout(btn_hlay)

    def create_unstructured_grid(self, node_table, element_table, X, show_grid, show_label):
        # offset = []  # Removed in the latest version of PyVista
        cell_type = []
        cells = []
        points = []
        node_label_dict = {'coordinates': [], 'labels': []}

        for elem in element_table:
            cell_p_num = 8  # For hexahedral element
            elem_type = vtk.VTK_HEXAHEDRON
            p_num_prev = len(points)

            cell_type.append(elem_type)

            p1 = [elem.node1.x, elem.node1.y, 0]
            p2 = [elem.node2.x, elem.node2.y, 0]
            p3 = [elem.node3.x, elem.node3.y, 0]
            p4 = [elem.node4.x, elem.node4.y, 0]
            p5 = [elem.node1.x, elem.node1.y, -elem.t]
            p6 = [elem.node2.x, elem.node2.y, -elem.t]
            p7 = [elem.node3.x, elem.node3.y, -elem.t]
            p8 = [elem.node4.x, elem.node4.y, -elem.t]

            points += [p1, p2, p3, p4, p5, p6, p7, p8]
            cells += [cell_p_num, p_num_prev, p_num_prev+1, p_num_prev+2,
                      p_num_prev+3, p_num_prev+4, p_num_prev+5,
                      p_num_prev+6, p_num_prev+7]

        self.node_coordinate_array = np.zeros((len(node_table), 2))  # For showing selected node in GUI
        for i_node, node in enumerate(node_table):
            self.node_coordinate_array[i_node, 0] = node.x
            self.node_coordinate_array[i_node, 1] = node.y

            if show_label:
                if node.dof_x_constr or node.dof_y_constr:
                    node_label_dict['coordinates'].append([node.x, node.y, 0])
                    if node.dof_x_constr and node.dof_y_constr:
                        node_label_dict['labels'].append('{:.0f} [X,Y constr]'.format(node.id))
                    elif node.dof_x_constr:
                        node_label_dict['labels'].append('{:.0f} [X constr]'.format(node.id))
                    elif node.dof_y_constr:
                        node_label_dict['labels'].append('{:.0f} [Y constr]'.format(node.id))
                if node.dof_x_load:
                    node_label_dict['coordinates'].append([node.x, node.y, 0])
                    node_label_dict['labels'].append('{:.0f} [X load={}]'.format(node.id, node.dof_x_load))
                if node.dof_y_load:
                    node_label_dict['coordinates'].append([node.x, node.y, 0])
                    node_label_dict['labels'].append('{:.0f} [Y load={}]'.format(node.id, node.dof_y_load))

        # For showing selected node number and its parent elements
        self.node_table = node_table
        self.node_selection_r = self.get_node_selection_radius(element_table)  # For showing selected node in GUI

        self.mesh = pv.UnstructuredGrid(np.array(cells),
                                        np.array(cell_type), np.array(points))
        if node_label_dict['labels']:
            self.plotter.add_point_labels(
                np.array(node_label_dict['coordinates']),
                node_label_dict['labels'], point_size=0, font_family='times', font_size=12
                )
        self.mesh.cell_data['X'] = X  # Set volfrac values

        # Colorbar
        sargs = dict(title='x',
                     interactive=False,
                     vertical=True,
                     height=0.2,
                     width=0.02,
                     position_x=0.02,
                     position_y=0.02,
                     title_font_size=16,
                     label_font_size=12,
                     font_family=FONT_FAMILY,
                     n_labels=4,
                     shadow=True)
        self.plotter.add_mesh(self.mesh, scalars='X', show_edges=True,
                              color='white', opacity=1, scalar_bar_args=sargs,
                              clim=[0, 1], ambient=0.5)
        if not VISTA_SHOW_COLORBAR:
            self.plotter.remove_scalar_bar()
        self.plotter.enable_depth_peeling(10)
        self.plotter.reset_camera()
        if show_grid:
            self.plotter.show_bounds(xtitle='X-axis',
                                     ytitle='Y-axis',
                                     n_xlabels=5,
                                     n_ylabels=5,
                                     bold=False,
                                     location='outer',
                                     ticks='inside',
                                     fmt='{:.2f}',
                                     minor_ticks=False,
                                     font_family=FONT_FAMILY,
                                     font_size=VISTA_LABEL_FONTSIZE,
                                     show_zaxis=False,
                                     show_zlabels=False)

        self.plotter.view_xy()
        self.plotter.track_click_position(callback=self.print_selected_node, side='left', viewport=False)
        self.text_actor = self.plotter.add_text('No node selected', position='upper_left', font='times', font_size=7)

    def take_screenshot(self):
        self.plotter.screenshot(f'screenshot.{VISTA_SCREENSHOT_FORMAT}',
                                transparent_background=False, scale=1,
                                window_size=VISTA_SCREENSHOT_SIZE)

    def print_selected_node(self, pos):
        self.plotter.remove_actor(self.text_actor)
        selected_node_index = np.argwhere(np.logical_and(abs(self.node_coordinate_array[:,0]-pos[0])<self.node_selection_r,
                                                         abs(self.node_coordinate_array[:,1]-pos[1])<self.node_selection_r))

        selected_node_num = len(selected_node_index)
        if selected_node_num == 0:
            self.text_actor = self.plotter.add_text('Mouse pos: X={:.3f} Y={:.3f}\nNo node selected'.format(pos[0], pos[1]),
                                                    position='upper_left', font='times', font_size=7)
        elif selected_node_num == 1:
            selected_node_id = selected_node_index[0, 0] + 1
            parent_elements = self.node_table[selected_node_index[0, 0]].parent_elem_id_lst

            self.text_actor = self.plotter.add_text('Mouse pos: X={:.3f} Y={:.3f}\nSelected node: {:.0f},  pos: X={:.3f} Y={:.3f}\nParent element(s): {}'.format(pos[0], pos[1], int(selected_node_index[0, 0]+1),
                                                    float(self.node_coordinate_array[selected_node_index[0, 0], 0]), float(self.node_coordinate_array[selected_node_index[0, 0], 1]), parent_elements),
                                                    position='upper_left', font='times', font_size=7)
        else:
            self.text_actor = self.plotter.add_text('Mouse pos: X={:.3f} Y={:.3f}\nMore than one node selected'.format(pos[0], pos[1]),
                                                    position='upper_left', font='times', font_size=7)

    def xy_view(self):
        self.plotter.view_xy()

    def isometric_view(self):
        self.plotter.view_isometric()

    @staticmethod
    def get_node_selection_radius(element_table):
        total_2d_area = 0.0
        for elem in element_table:
            elem.calc_area()
            total_2d_area += elem.area

        r = np.sqrt(total_2d_area/(len(element_table)))

        return r/4


class CreateProjWindow(QtWidgets.QDialog):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Create Project')

        main_vlay = QtWidgets.QVBoxLayout()

        glay = QtWidgets.QGridLayout()

        pname_label = QtWidgets.QLabel('Project Name')
        ploc_label = QtWidgets.QLabel('Project Location')
        inputloc_label = QtWidgets.QLabel('Input File Location')

        self.pname_lineedit = QtWidgets.QLineEdit()
        self.ploc_lineedit = QtWidgets.QLineEdit()
        self.inputloc_lineedit = QtWidgets.QLineEdit()

        ploc_btn = QtWidgets.QPushButton('Browse')
        ploc_btn.clicked.connect(self.browse_project)
        inputloc_btn = QtWidgets.QPushButton('Browse')
        inputloc_btn.clicked.connect(self.browse_inputfile)

        glay.addWidget(pname_label, 0, 0)
        glay.addWidget(ploc_label, 1, 0)
        glay.addWidget(inputloc_label, 2, 0)
        glay.addWidget(self.pname_lineedit, 0, 1)
        glay.addWidget(self.ploc_lineedit, 1, 1)
        glay.addWidget(self.inputloc_lineedit, 2, 1)
        glay.addWidget(ploc_btn, 1, 2)
        glay.addWidget(inputloc_btn, 2, 2)

        hlay = QtWidgets.QHBoxLayout()

        cancel_btn = QtWidgets.QPushButton('Cancel')
        cancel_btn.clicked.connect(self.close)
        self.create_btn = QtWidgets.QPushButton('Create')

        hlay.addStretch(1)
        hlay.addWidget(cancel_btn)
        hlay.addWidget(self.create_btn)

        main_vlay.addLayout(glay)
        main_vlay.addStretch()
        main_vlay.addLayout(hlay)

        self.setLayout(main_vlay)

        self.show()

    def browse_inputfile(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Select Input File", "","Input File (*.txt)")
        if file_name:
            self.inputloc_lineedit.setText(file_name)

    def browse_project(self):
        directory_name = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Project Folder"))
        if directory_name:
            self.ploc_lineedit.setText(directory_name)


class FilterElements(QtWidgets.QDialog):

    def __init__(self, struct, file_index, statusBar):
        super().__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)

        self.resize(700, 500)
        self.setWindowTitle('Filter Elements')

        self.struct = struct
        self.struct_new = None
        self.file_index = file_index
        self.filter_thresh_default = 0.8

        self.add_buttons()
        self.init_mesh()

        self.setLayout(self.main_vlay)

        self.statusBar = statusBar

    def add_buttons(self):
        self.main_vlay = QtWidgets.QVBoxLayout()

        btn_hlay = QtWidgets.QHBoxLayout()
        btn_hlay.setContentsMargins(10, 0, 10, 0)

        filter_btn = QtWidgets.QPushButton('Filter')
        filter_btn.setAutoDefault(False)
        filter_btn.clicked.connect(self.filter_elem)

        filter_thresh_label = QtWidgets.QLabel('Filter Threshold:')
        self.filter_thresh_dspin = QtWidgets.QDoubleSpinBox()
        self.filter_thresh_dspin.setMinimum(0.0)
        self.filter_thresh_dspin.setMaximum(1.0)
        self.filter_thresh_dspin.setDecimals(3)
        self.filter_thresh_dspin.setSingleStep(0.001)
        self.filter_thresh_dspin.setValue(self.filter_thresh_default)

        screenshot_btn = QtWidgets.QPushButton('Screenshot')
        screenshot_btn.setAutoDefault(False)
        screenshot_btn.clicked.connect(self.take_screenshot)

        export_inp_btn = QtWidgets.QPushButton('Export INP')
        export_inp_btn.setAutoDefault(False)
        export_inp_btn.clicked.connect(self.export_mesh_inp)

        export_stl_btn = QtWidgets.QPushButton('Export STL')
        export_stl_btn.setAutoDefault(False)
        export_stl_btn.clicked.connect(self.export_mesh_stl)

        btn_hlay.addWidget(filter_btn)
        btn_hlay.addSpacing(20)
        btn_hlay.addWidget(filter_thresh_label)
        btn_hlay.addWidget(self.filter_thresh_dspin)
        btn_hlay.addStretch()
        btn_hlay.addWidget(screenshot_btn)
        btn_hlay.addWidget(export_inp_btn)
        btn_hlay.addWidget(export_stl_btn)

        self.main_vlay.addLayout(btn_hlay)

    def init_mesh(self):
        file_lst = MainWindow.get_files()
        self.bounding_box = self.struct.get_bounding_box()

        self.X = np.fromfile('X/'+file_lst[self.file_index], dtype=np.float64)

        self.draw_widget = VistaMesh(self.struct.node_table, self.struct.element_table, self.X, show_grid=False, show_label=False)

        self.main_vlay.addWidget(self.draw_widget)

    def filter_elem(self):
        self.struct_new = copy.copy(self.struct)
        min_Xe = self.filter_thresh_dspin.value()

        self.struct_new.set_X(self.X)
        self.struct_new.filter_out_elements(min_Xe=min_Xe)
        X_ = self.struct_new.get_X_array()

        if self.draw_widget != None:
            self.draw_widget.setParent(None)
            self.draw_widget = None

        self.draw_widget = VistaMesh(self.struct_new.node_table, self.struct_new.element_table, X_, show_grid=False, show_label=False)

        self.main_vlay.addWidget(self.draw_widget)

    def take_screenshot(self):
        self.draw_widget.take_screenshot()
        self.statusBar.showMessage('Screenshot taken. Please check the project directory.')

    def export_mesh_inp(self):
        if not self.struct_new:
            self.statusBar.showMessage('Please click "Filter" button first.')
            return
        self.struct_new.write_mesh(output_filename='opt.inp', mesh_type='quad', write_format='abaqus')
        self.statusBar.showMessage('Exported to opt.inp file. Please check the project directory.')

    def export_mesh_stl(self):
        if not self.struct_new:
            self.statusBar.showMessage('Please click "Filter" button first.')
            return
        self.struct_new.write_mesh(output_filename='opt.stl', mesh_type='triangle', write_format='stl')
        self.statusBar.showMessage('Exported to opt.stl file. Please check the project directory.')


class PlotCompliance(QtWidgets.QDialog):

    def __init__(self):
        super().__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)

        self.resize(700, 500)
        self.setWindowTitle('Compliance History Over Iterations')

        refresh_btn = QtWidgets.QPushButton('Refresh')
        refresh_btn.setAutoDefault(False)
        refresh_btn.clicked.connect(self.refresh_compl_plot)

        hlay = QtWidgets.QHBoxLayout()
        hlay.addWidget(refresh_btn)
        hlay.addStretch()

        self.main_lay = QtWidgets.QVBoxLayout()
        self.main_lay.addLayout(hlay)
        self.main_lay.addWidget(self.plot_compl())

        self.setLayout(self.main_lay)

    def refresh_compl_plot(self):
        self.canvas.setParent(None)
        self.main_lay.addWidget(self.plot_compl())

    def plot_compl(self):
        fig, ax = plt.subplots(figsize=MATPLOT_FIGSIZE)

        try:
            dat = np.loadtxt('compl_plot.txt')
            dat = np.transpose(dat)

            plt.plot(dat[0], dat[1], linewidth=1, color='#6C8EBF')
            plt.xlabel('Iteration', fontsize=MATPLOT_AXES_LABEL_FONTSIZE, weight='bold', color=MATPLOT_AXES_LABEL_COLOR)
            plt.ylabel('Compliance', fontsize=MATPLOT_AXES_LABEL_FONTSIZE, weight='bold', color=MATPLOT_AXES_LABEL_COLOR)
            plt.xticks(fontsize=MATPLOT_TICKS_LABEL_FONTSIZE)
            plt.yticks(fontsize=MATPLOT_TICKS_LABEL_FONTSIZE)
            ax.yaxis.get_ticklocs(minor=True)
            ax.xaxis.get_ticklocs(minor=True)
            ax.minorticks_on()
            plt.savefig(f'compl_plot.{MATPLOT_EXPORT_FORMAT}', dpi=DPI)
            self.canvas = FigureCanvasQTAgg(fig)

            return self.canvas
        except Exception as e:
            print(e)
            label = QtWidgets.QLabel('Compliance data not generated yet.')

            return label


class PlotMaxXChange(QtWidgets.QDialog):

    def __init__(self):
        super().__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowMaximizeButtonHint)

        self.resize(700, 500)
        self.setWindowTitle('Max. Relative Material Density Change History Over Iterations')

        refresh_btn = QtWidgets.QPushButton('Refresh')
        refresh_btn.setAutoDefault(False)
        refresh_btn.clicked.connect(self.refresh_max_xchange_plot)

        hlay = QtWidgets.QHBoxLayout()
        hlay.addWidget(refresh_btn)
        hlay.addStretch()

        self.main_lay = QtWidgets.QVBoxLayout()
        self.main_lay.addLayout(hlay)
        self.main_lay.addWidget(self.plot_max_xchange())

        self.setLayout(self.main_lay)

    def refresh_max_xchange_plot(self):
        self.canvas.setParent(None)
        self.main_lay.addWidget(self.plot_max_xchange())

    def plot_max_xchange(self):
        fig, ax = plt.subplots(figsize=MATPLOT_FIGSIZE)

        try:
            dat = np.loadtxt('max_xchange_plot.txt')
            dat = np.transpose(dat)

            plt.plot(dat[0], dat[1], linewidth=1, color='#B58300')
            plt.xlabel('Iteration', fontsize=MATPLOT_AXES_LABEL_FONTSIZE, weight='bold', color=MATPLOT_AXES_LABEL_COLOR)
            plt.ylabel(r'Max. change of $x_e$', fontsize=MATPLOT_AXES_LABEL_FONTSIZE, weight='bold', color=MATPLOT_AXES_LABEL_COLOR)
            plt.xticks(fontsize=MATPLOT_TICKS_LABEL_FONTSIZE)
            plt.yticks(fontsize=MATPLOT_TICKS_LABEL_FONTSIZE)
            # plt.grid(color='gray', linestyle=':', linewidth=0.5)
            plt.ylim(0, 1)
            ax.yaxis.get_ticklocs(minor=True)
            ax.xaxis.get_ticklocs(minor=True)
            ax.minorticks_on()
            plt.savefig(f'max_xchange_plot.{MATPLOT_EXPORT_FORMAT}', dpi=DPI)
            self.canvas = FigureCanvasQTAgg(fig)

            return self.canvas
        except Exception as e:
            print(e)
            label = QtWidgets.QLabel('Max. relative material density change data not generated yet.')

            return label


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion")
    app.setWindowIcon(QtGui.QIcon('icon.svg'))

    mw = MainWindow()
    mw.showMaximized()

    sys.exit(app.exec_())
