import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy import ndimage
from sunpy.coordinates import SphericalScreen

"""
This piece of code is mainly a class called Rectangle which is used to produce a time-distance plot using this rectangle.
The time-distance plot is produce with distance being along the length of the rectangle: pixels are summed along the width of the rectangle.
This has been tested with AIA cutouts as input for the maps.

How does it work?
1. The user provides the input to define the Rectangle
2. To produce the time distance plot, we are calculating the "colomns" (for each time) one by one, using maps in a sunpy map sequence. We loop on the maps.
3. For each map then, we use the rotation tool of ndimage to rotate the AIA cutout so that the box is aligned with the (x,y) axes: in that rotated frame we can select pixels in one direction and sum pixels in the other direction
4. In this rotated frame, we calculate one colomn of the final time-distance plot
5. We do this for each map in the map sequence, and then put it together in one array.
"""

def coordinates_in_rotated_frame(px,py,rotation_angle,array):
    """
    This function is needed to recalculate the coordinates of the box in the rotated frame

    Inputs
        px, py: pixel coordinates of a point (usually the rectangle center) in the image before rotation
        rotation_angle: quantity (with units) of the angle to apply for the rotation
        array: original data array (before rotation)

    Output
        new_px, new_py: pixel coordinates in the rotated frame
    """
    
    # extract the width and length of the data array
    Lx = np.shape(array)[1]
    Ly = np.shape(array)[0]

    # calculate polar coordinates before rotation
    center_r = np.sqrt(px*px+py*py)
    center_angle = np.arctan(py/px) * u.rad
    
    # polar coordinate in (intermediary) rotated frame 
    new_center_r = center_r
    new_angle = center_angle - rotation_angle 

    # shift in x and y for the final frame
    if (rotation_angle>=0)&(rotation_angle<=90*u.deg):
        x_shift = 0
        y_shift = Lx*np.sin(rotation_angle)
    if (rotation_angle>90*u.deg)&(rotation_angle<=180*u.deg):
        x_shift = Lx*np.abs(np.cos(rotation_angle))
        y_shift = Lx*np.abs(np.sin(rotation_angle)) + Ly*np.abs(np.cos(rotation_angle))
    if (rotation_angle>=-90*u.deg)&(rotation_angle<0*u.deg):
        x_shift = Ly*np.abs(np.sin(rotation_angle))
        y_shift = 0
    if (rotation_angle>-180*u.deg)&(rotation_angle<-90*u.deg):
        x_shift = Lx*np.abs(np.cos(rotation_angle)) + Ly*np.abs(np.sin(rotation_angle))
        y_shift = Ly*np.abs(np.cos(rotation_angle))
    
    #new coordinates
    new_px = new_center_r*np.cos(new_angle) + x_shift
    new_py = new_center_r*np.sin(new_angle) + y_shift

    return np.array([new_px,new_py])

class Rectangle:
    def __init__(self, center, length, width, angle):
        """
        The inputs are quantities (they have units). Example of valid inputs:
            center = [995*u.arcsec,-290*u.arcsec]   ## in arcsec
            length = 120*u.arcsec                   ## in arcsec
            width = 30*u.arcsec                     ## in arcsec
            angle = Angle(-15, u.degree)            ## in degrees
        """
        self.center = center
        self.length = length
        self.width = width
        self.angle = angle

    def get_corners(self):
        """
        This function returns the coordinates of the four corners of the rectangle (quantities)
        """
        x1 = self.center[0] - self.length/2.*np.cos(self.angle.radian) - self.width/2*np.sin(self.angle.radian)
        #x2 = self.center[0] - self.length/2.*np.cos(self.angle.radian) + self.width/2*np.sin(self.angle.radian)
        y1 = self.center[1] - self.length/2.*np.sin(self.angle.radian) + self.width/2*np.cos(self.angle.radian)
        #y2 = self.center[1] - self.length/2.*np.sin(self.angle.radian) - self.width/2*np.cos(self.angle.radian)
        dx = self.length*np.cos(self.angle.radian)
        dy = self.length*np.sin(self.angle.radian)
        dxp = self.width*np.sin(self.angle.radian)
        dyp = self.width*np.cos(self.angle.radian)
        corner1 = {'x':x1, 'y':y1}
        corner2 = {'x':x1+dx, 'y':y1+dy}
        corner3 = {'x':x1+dxp, 'y':y1-dyp}
        corner4 = {'x':x1+dx+dxp, 'y':y1+dy-dyp}
        return [corner1,corner2,corner3,corner4]
    
    def get_unitless_lines(self):
        """
        This function retunrs the coordinates of the lines that represents the sides of the rectangle, without their units (floats).
        The units are removed as these lines are meant to be used to plot the rectangle with the next function.
        """
        corners = self.get_corners()
        line1x = np.array([corners[0]['x'].value, corners[1]['x'].value])
        line1y = np.array([corners[0]['y'].value, corners[1]['y'].value])
        line2x = np.array([corners[2]['x'].value, corners[3]['x'].value]) 
        line2y = np.array([corners[2]['y'].value, corners[3]['y'].value])
        line3x = np.array([corners[0]['x'].value, corners[2]['x'].value])
        line3y = np.array([corners[0]['y'].value, corners[2]['y'].value])
        line4x = np.array([corners[1]['x'].value, corners[3]['x'].value])
        line4y = np.array([corners[1]['y'].value, corners[3]['y'].value]) 
        line1 = {'x':line1x, 'y':line1y}
        line2 = {'x':line2x, 'y':line2y}
        line3 = {'x':line3x, 'y':line3y}
        line4 = {'x':line4x, 'y':line4y}
        return [line1, line2, line3, line4]
    
    def plot(self, ax):
        """
        This function plots the rectangle. I should allow for the user to change the keywords --- to be done later
        """
        lines = self.get_unitless_lines()
        for line in lines:
            ax.plot(line['x']*u.arcsec.to(u.deg), line['y']*u.arcsec.to(u.deg),
                    color='white', linewidth=0.5,
                    transform=ax.get_transform("world"))
    
    def get_center_coordinate_at_given_time(self, obstime):
        """
        This function returns the SkyCoord coordinates of the rectangle center. This requires to give an observation time.
        The frame is set to helioprojective, and the observer to Earth - as I only used this script on AIA data...
        At a later date I should modify this so that the user can change those too - or better that they provide a map and that the observer is found from there? --- TBD
        """
        rectangle_center_coord = SkyCoord(self.center[0], self.center[1], 
                                  obstime=obstime, 
                                  observer="earth", 
                                  frame="helioprojective")
        return rectangle_center_coord
    
    def get_distance_line_from_map(self, map):
        """
        This function takes a map and calculate the "time-distance" line (or column) corresponding to this map.
        To do so it does the following:
        - Rotate the data array to align with the box direction
        - Do some pixel calculation along the rotation
        - Extract the data within the rectangle in a new array, and sum pixels on one dimension (representing the width of the box)
        - Return the one-dimension array (the line/column of the time-distance plot being constructed)

        Inputs:
            A sunpy map

        Output: 
            line: one-dimension array representing the intensity of pixels along the length of the box (summed along the width)
            distance_axis: one-dimension array with the distance information in physical units (arcsec)
        """
        ### Extract information from the map
        data_array = map.data
        pixel_scale = 0.5*(map.scale[0]+map.scale[1])
      
        ### Calculate box information in pixels
        rectangle_width_inpix = self.width/pixel_scale
        rectangle_length_inpix = self.length/pixel_scale
        rectangle_center_coord = self.get_center_coordinate_at_given_time(map.date)
     
        px, py = map.wcs.world_to_pixel(rectangle_center_coord)
        if np.isnan(px) or np.isnan(py):
            with SphericalScreen(map.observer_coordinate):
                px, py = map.wcs.world_to_pixel(rectangle_center_coord)

        ### rotate data
        rotation_angle = self.angle
        rotated_data = ndimage.rotate(data_array, rotation_angle, reshape=True)
    
        ### calculate coordinates of the box in the rotated data
        new_coords = coordinates_in_rotated_frame(px,py,rotation_angle,data_array)
        rectangle_xx = new_coords[0] + 0.5*rectangle_length_inpix.value*np.array([-1,1])
        rectangle_yy = new_coords[1] + 0.5*rectangle_width_inpix.value*np.array([-1,1])
    
        ### extract data
        in_box = rotated_data[ int(rectangle_yy[0]):int(rectangle_yy[1]), int(rectangle_xx[0]):int(rectangle_xx[1])]
        line = np.sum(in_box,0)
    
        ## create a array for the axis
        distance_axis = np.squeeze(np.arange(len(line))*u.pixel*pixel_scale)
    
        return line, distance_axis
    
    def get_time_distance_array(self, map_sequence):
        """
        This a wrapper function that will loop through maps of a map sequence to populate the array "time_distance"

        Input:
            sunpy map sequence

        Output:
            dict containing the time-distance array, and the associated time and distance arrays (time in datetime format and distance as quantities (with units))
        """
        time_distance = []
        time = []
        for map in map_sequence:
            line, distance = self.get_distance_line_from_map(map)
            time_distance.append(line)
            time.append(map.date.to_datetime())
        time_distance_array=np.array(time_distance).transpose()
        time_array = np.array(time)
        return {'time-distance': time_distance_array, 'time': time_array, 'distance': distance}