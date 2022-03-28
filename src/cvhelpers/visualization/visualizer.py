"""
Easy visualization of point clouds and geometric primitives
Code adapted from vtk_visualizer

   Original Author: Øystein Skotheim, SINTEF ICT <oystein.skotheim@sintef.no>
   Date:   Thu Sep 12 15:50:40 2013


Example:

>>> import numpy as np
>>> import open3d as o3d
>>> from cvhelpers.visualization.visualizer import Visualizer

>>> vtk_control = Visualizer()
>>> cloud = o3d.io.read_point_cloud('cloud.ply')

>>> cloud_obj = vtk_control.create_point_cloud(np.asarray(cloud.points))
>>> vtk_control.add_object(cloud_obj)
>>> vtk_control.start()

"""
import glob
import logging
import math
import json
import os
import re
from typing import Optional, List

import numpy as np
import vtk

from .vtk_object import VTKObject

_CAM_JSON_PATH = 'viz_cameras.json'


class Visualizer:
    """Helper class for easier visualization of point clouds and geometric primitives"""

    # noinspection PyArgumentList
    def __init__(self, bg_color=None, win_size=(1024, 768),
                 num_renderers:int = 1, viewports: Optional[List[List[float]]] = None,
                 share_camera: bool = True):
        """Create a wiget with a VTK Visualizer Control in it

        Args:
            bg_color: RGB color of background, numbers in range [0.0, 1.0]. Default: Black
            num_renderers: Number of renderers to put into the same window
            viewports: Defines the viewport range of each renderer. If provided, should
              have num_renderers elements, with each being a tuple/list containing
              [xmin, ymin, xmax, ymax]
            share_camera: Whether to share the camera between renderers
        """
        self.render_window = vtk.vtkRenderWindow()
        self.renderers = []

        # Decide viewport ranges
        if viewports is None:
            viewports = self._compute_default_viewports(num_renderers)

        for i in range(num_renderers):
            self.renderers.append(vtk.vtkRenderer())
            self.renderers[-1].SetViewport(*viewports[i])
            self.render_window.AddRenderer(self.renderers[-1])
            if bg_color is not None:
                self.renderers[i].SetBackground(*bg_color)
            if share_camera and i > 0:
                self.renderers[-1].SetActiveCamera(self.renderers[0].GetActiveCamera())

        self.render_window.SetSize(win_size[0], win_size[1])
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetInteractorStyle(InteractorStyle(self))
        self.iren.RemoveObservers('CharEvent')
        self.iren.SetRenderWindow(self.render_window)
        self.default_renderer = self.renderers[0]
        self._logger = logging.getLogger(self.__class__.__name__)

        # For resizing points
        self._objects = []
        self._labelObjects = []  # for showing plot titles
        self._share_camera = share_camera

    @staticmethod
    def _compute_default_viewports(num_renderers):
        if num_renderers == 1:
            viewports = [[0.0, 0.0, 1.0, 1.0]]
        else:
            # Try to ensure roughly the same columns and rows
            ncols = math.ceil(math.sqrt(num_renderers))
            nrows = math.ceil(num_renderers / ncols)
            viewports = []
            for i in range(num_renderers):
                icol = i % ncols
                irow = nrows - 1 - (i // ncols)
                xmin = icol / ncols
                xmax = (icol + 1) / ncols
                ymin = irow / nrows
                ymax = (irow + 1) / nrows
                viewports.append([xmin, ymin, xmax, ymax])
        return viewports

    def save_cameras(self):
        """Save camera parameters to viz_cameras.json"""
        cam_params = []
        for iren in range(len(self.renderers)):
            cam = self.renderers[iren].GetActiveCamera()
            cam_param = {
                'Position': cam.GetPosition(),
                'FocalPoint': cam.GetFocalPoint(),
                'ViewUp': cam.GetViewUp(),
                'ViewAngle': cam.GetViewAngle(),
                'ClippingRange': cam.GetClippingRange()
            }
            cam_params.append(cam_param)

        with open(_CAM_JSON_PATH, 'w') as fid:
            json.dump(cam_params, fid, indent=2)
            self._logger.info('Saved camera parameters to {}'.format(_CAM_JSON_PATH))

    def restore_cameras(self):
        """Restore camera parameters from viz_cameras.json"""
        if not os.path.exists(_CAM_JSON_PATH):
            self._logger.error('Json config file not found')
            return

        with open(_CAM_JSON_PATH, 'r') as fid:
            cam_params = json.load(fid)
            if len(cam_params) != len(self.renderers):
                self._logger.error('Json files does not contain the same number of cameras')
                return

            for iren in range(len(self.renderers)):
                cam = self.renderers[iren].GetActiveCamera()
                cam.SetPosition(cam_params[iren]['Position'])
                cam.SetFocalPoint(cam_params[iren]['FocalPoint'])
                cam.SetViewUp(cam_params[iren]['ViewUp'])
                cam.SetViewAngle(cam_params[iren]['ViewAngle'])
                cam.SetClippingRange(cam_params[iren]['ClippingRange'])

        self._logger.info('Restored camera parameters from {}'.format(_CAM_JSON_PATH))

    def increase_point_size(self):
        for obj in self._objects:
            actor = obj.actor
            if actor.GetProperty().GetRepresentation() == vtk.VTK_POINTS:
                prev_size = actor.GetProperty().GetPointSize()
                actor.GetProperty().SetPointSize(prev_size * 1.5)

        self.default_renderer.GetRenderWindow().Render()

    def decrease_point_size(self):
        for obj in self._objects:
            actor = obj.actor
            if actor.GetProperty().GetRepresentation() == vtk.VTK_POINTS:
                prev_size = actor.GetProperty().GetPointSize()
                actor.GetProperty().SetPointSize(prev_size / 1.5)

        self.default_renderer.GetRenderWindow().Render()

    def show_hide_object(self, obj_ind):
        if obj_ind < len(self._objects):
            obj = self._objects[obj_ind]
            obj.GetActor().SetVisibility(1 if obj.GetActor().GetVisibility() == 0 else 0)
            self.default_renderer.GetRenderWindow().Render()
        else:
            self._logger.warning('Ignoring show_hide_object() with ind:{} as there are only {} objects'.format(
                obj_ind, len(self._objects)))

    def close_window(self):
        self.render_window.Finalize()
        self.iren.TerminateApp()
        del self.render_window, self.iren

    def add_object(self, obj, renderer_idx=0, visible=True):
        """Add a supplied vtkActor object to the visualizer"""
        if not visible:
            obj.GetActor().SetVisibility(0)  # Adds invisible object
        self._objects.append(obj)
        self.renderers[renderer_idx].AddActor(self._objects[-1].GetActor())

    def set_titles(self, titles):
        # Remove previous label (if any)
        for i in range(len(self._labelObjects)):
            self.renderers[i].RemoveActor(self._labelObjects[i].GetActor())
        self._labelObjects = []

        for i in range(len(titles)):
            obj = VTKObject()
            obj.CreateText(titles[i])
            self._labelObjects.append(obj)
            self.renderers[i].AddActor(self._labelObjects[-1].GetActor())

    def reset_camera(self):
        """Reset the cameras to fit contents"""
        if self._share_camera:
            self.renderers[0].ResetCamera()
        else:
            for iren in range(len(self.renderers)):
                self.renderers[iren].ResetCamera()
        self.render()

    def render(self):
        """Render all objects"""
        self.render_window.Render()

    def start(self):
        """Run event loop"""
        self.iren.Start()

    def set_window_background(self, r, g, b):
        """Set the background color of the visualizer to given R, G and B color"""
        for iren in range(len(self.renderers)):
            self.renderers[iren].SetBackground(r, g, b)

    def save_screenshot(self, filename=None):
        """Takes a screenshot of the visualizer in png format"""

        def _get_next_filename():
            already_present = glob.glob('screenshots/*.png')
            if len(already_present) == 0:
                return 'screenshots/0.png'

            max_index = -1
            for p in already_present:
                index = re.findall(r'\d+', p)
                if len(index) == 1:
                    max_index = max(max_index, int(index[0]))

            return 'screenshots/{}.png'.format(max_index + 1)

        if filename is None:
            if not os.path.isdir('screenshots'):
                os.makedirs('screenshots')
            filename = _get_next_filename()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        self._logger.info('Saved screenshot to {}'.format(filename))


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Extends the default vtkInteractorStyleTrackballCamera to support
    keypresses for common operations:
    - '+'/'-': Make point sizes larger or smaller
    """
    def __init__(self, vis: Visualizer):
        """
        Args:
            vis: Instance of visualizer
        """
        super().__init__()
        self.vis = vis
        self.AddObserver("KeyPressEvent", self._key_press_event)

    def _key_press_event(self, obj, event):
        """Handle resizing of points"""
        key = self.GetInteractor().GetKeySym()
        if key == 'KP_Add' or key == 'plus' or key == 'equal':
            self.vis.increase_point_size()
        elif key == 'KP_Subtract' or key == 'minus':
            self.vis.decrease_point_size()
        elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            obj_ind = (int(key) - 1) % 10
            self.vis.show_hide_object(obj_ind)
        elif key.lower() == 'q':
            self.vis.close_window()
        elif key.lower() == 'r':
            self.vis.reset_camera()
