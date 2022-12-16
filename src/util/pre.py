#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:41:53 2022

@author: jhautala
"""

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline
import skimage

# intra-project
from util.model import AnnoImg, landmark68, cheeks, h_syms, nose_i, outer_canthi


# some constants for coord math (mostly premature optimization)
pi_2 = np.pi / 2.
to_deg = 180. / np.pi
to_rad = np.pi / 180.
id_2d = np.array([[1, 0], [0, 1]])


class Sym:
    def __init__(self, desc, pairs, weight=1.):
        self.desc = desc
        self.pairs = np.array(pairs)
        self.weight = weight
        self.weight_tot = len(self.pairs) * weight

    def get_left(self):
        return self.pairs[:, 0]

    def get_right(self):
        return self.pairs[:, 1]


default_syms = [
    # NOTE: I haven't tested this a lot, but I don't expect
    #       it will be a great/reliable way to address tilt...
    # Sym(
    #     'cheeks',
    #     ((i, 16 - i) for i in range(8)),
    #     .5,
    # ),
    # NOTE: The corners of the eyes seem good-enough on their own
    #       for the purpose of normalizing yaw
    Sym(
        'canthi',
        [
            outer_canthi,
            [39, 42],
        ],
        4.  # higher weight for canthi
    ),
    # TODO: delete?
    # NOTE: This is probably not worth keeping, especially
    #       since its weight is so low...
    Sym(
        'eyelids',
        [
            # [36, 45], # already included in eyes corners
            [37, 44],
            [38, 43],
            # [39, 42], # already included in eyes corners
            [40, 47],
            [41, 46],
        ],
        .5,  # due to expressions (e.g. squinting)
    ),
]


def _get_yaw(coords, sym):
    '''
    This function calculates the angle offset based on the given
    coordinates and expected symmetric point pairs.

    Parameters
    ----------
    coords : number with shape(landmarks,dimensions)
        the landmark coordinates to check.
    sym : Sym
        The basis of symmetry (essentially a combination of point pairs).

    Returns
    -------
    angle : float
        The estimated angle of rotation (in radians).

    '''
    # calculate diffs per pair
    left, right = np.squeeze(np.split(coords[sym.pairs.T], 2))
    xx, yy = np.squeeze(np.split((right - left).T, 2))
    yy = -yy  # neg y because image y starts at top

    # calculate pair distances
    hypots = np.sqrt(xx ** 2 + yy ** 2)
    weight_tot = np.sum(hypots)  # use pair distance as weight

    x = np.sum(xx * hypots) / weight_tot
    y = np.sum(yy * hypots) / weight_tot

    # calculate angle
    y_neg = np.sign(y) == -1
    if x == 0:
        angle = -pi_2 if y_neg else pi_2
    else:
        angle = np.arctan(y / x)
        x_neg = np.sign(x) == -1
        if x_neg:
            if y_neg:
                angle -= np.pi
            else:
                angle += np.pi

    return angle


def get_yaw(anno, syms=default_syms, deg=False):
    coords = anno.get_coords()
    angles = []
    weights = []
    for sym in syms:
        angle = _get_yaw(coords, sym)
        angles.append(angle)
        weights.append(sym.weight_tot)
    angles = np.array(angles)
    weights = np.array(weights)
    angle = np.sum(angles * weights) / np.sum(weights)
    if deg:
        return to_deg * angle
    else:
        return angle


def get_yaw_data(anno):
    '''
    Extract image metadata associated with center/rotate logic.

    Parameters
    ----------
    anno : AnnoImg
        the image to analyze.

    Returns
    -------
    in_dims 2-item array of number
        original image dimensions.
    margin 2-item array of number
        margin size to fit features such that in_dims + 2*margin = out_dims.
    face : array of number
        original coordinates of the center of the face.
    angle : number
        estimated angle of rotation (in radians).
    coords : number(68,2)
        new landmark coordinates, rotated and centered.

    '''
    # anno = cache.get_image(118)

    # get image and calculate translation
    img = anno.get_image()
    face = anno.get_face_center()
    center = [np.nan, np.nan] if img is None else np.array([[img.width / 2, img.height / 2]])
    coords = anno.get_coords()

    # calculate rotation
    angle = get_yaw(anno)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotx = np.array([[cos, sin], [-sin, cos]])

    # perform rotation
    coords = coords - face  # move coordinates to the origin to rotate
    coords = coords @ rotx  # apply rotation matrix
    if img is not None:
        coords = coords + center  # move coordinates to image center
    else:
        # NOTE: This is currently only here support the case where raw image data is unavailable...
        #       Unfortunately, it complicates user expectations for this function
        coords = coords + face  # move coordinates back to original location

    # calculate buffer
    mins = np.amin(coords, axis=0)
    maxs = np.amax(coords, axis=0)
    if img is None:
        in_dims = np.array([np.nan, np.nan])
        margin = np.array([np.nan, np.nan])
    else:
        in_dims = np.array([img.width, img.height])
        margin = []
        for i in range(2):
            max_buff = 0
            lo_buff = 0 - mins[i]
            if lo_buff > max_buff:
                max_buff = lo_buff
            hi_buff = maxs[i] - in_dims[i]
            if hi_buff > max_buff:
                max_buff = hi_buff
            margin.append(max_buff + max_buff if max_buff > 0 else 0)

        margin = np.array(margin)
        coords += margin
    return in_dims, margin, face, angle, coords


def rotate(anno):
    def _img():
        img = anno.get_image()
        in_dims, margin, face, angle, coords = get_yaw_data(anno)

        # NOTE: We add a buffer around the image
        #       to avoid cropping content during rotating/centering
        buff = Image.new(img.mode, (img.width * 3, img.height * 3))
        buff.paste(img, (img.width, img.height))
        buff_face = face + np.array([[img.width, img.height]])
        buff_cen = np.array([[buff.width / 2, buff.height / 2]])
        cen = buff.transform(
            buff.size,
            method=Image.Transform.AFFINE,
            data=np.append(id_2d, (buff_face - buff_cen).T, 1).ravel(),
        )

        angle_deg = angle * to_deg
        rot = cen.rotate(
            -angle_deg,
            center=tuple(buff_cen.ravel()),
        )

        # crop back to original size
        cropped = rot.crop(
            (
                img.width - margin[0],
                img.height - margin[1],
                img.width * 2 + margin[0],
                img.height * 2 + margin[1]
            )
        )

        return cropped

    _, _, _, _, coords = get_yaw_data(anno)

    # add 'rotated' to the image description
    desc = anno.desc.copy()
    desc.append('rotated')
    return AnnoImg(
        anno.image_set,
        anno.filename,
        coords,
        _img,
        anno.row_id,
        desc=desc
    )


def crop(anno, use_splines=False):
    """
    Crop an image to a convex hull around landmarks.

    Parameters
    ----------
    anno : AnnoImg
        The image to crop.
    use_splines : bool
        True to interpolate lanmarks with splines. The default is False.

    Returns
    -------
    cropped : AnnoImge
        The cropped image.
    """
    def _img():
        image = np.array(anno.get_image())  # convert to skimage
        coords = anno.get_coords()
        if use_splines:
            X = []
            Y = []
            for f in landmark68.features:
                # f = landmark68.features[0]
                data = coords[f.idx]
                if np.any(np.isnan(data)): continue;
                points = data.T
                [xx, yy] = points
                distance = np.cumsum(
                    np.sqrt(np.sum(
                        np.diff(points.T, axis=0) ** 2,
                        axis=1
                    ))
                )
                if not distance[-1]: continue;

                distance = np.insert(distance, 0, 0) / distance[-1]
                splines = [UnivariateSpline(distance, point, k=2, s=.2) for point in points]
                points_fitted = np.vstack(
                    [spline(np.linspace(0, 1, 64)) for spline in splines]
                )
                points_fitted.shape
                X.extend(points_fitted[0])
                Y.extend(points_fitted[1])

            data = np.stack([X, Y], axis=1)
        else:
            data = coords
        data = data[~np.isnan(data).any(axis=1), :]
        poly = data[ConvexHull(data).vertices]
        X, Y = skimage.draw.polygon(poly[:, 0], poly[:, 1])
        cropped = np.zeros(image.shape, dtype=np.uint8)
        cropped[Y, X] = image[Y, X]
        return Image.fromarray(cropped)

    desc = anno.desc.copy().append('cropped')
    return AnnoImg(
        anno.image_set,
        anno.filename,
        anno.get_coords(),
        _img,
        anno.row_id,
        desc=desc
    )


def add_derived(cache):
    df = cache.get_meta()

    widths = []
    heights = []
    margins = []
    angles = []
    cenrots = []
    raws = []
    faces = []
    cenrot_widths = []
    cenrot_heights = []

    # calculate rotation/etc for each image
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        raws.append(anno.get_coords())
        in_dims, margin, face, angle, cenrot = get_yaw_data(anno)
        faces.append(face)
        widths.append(in_dims[0])
        heights.append(in_dims[1])
        margins.append(margins)
        cenrot_widths.append(widths[-1] + margin[0])
        cenrot_heights.append(heights[-1] + margin[1])
        angles.append(angle)
        cenrots.append(cenrot)

    # convert to numpy
    raws = np.array(raws)
    cenrots = np.array(cenrots)

    # calculate extents of landmarks (raw)
    extents = np.amax(raws, axis=1) - np.amin(raws, axis=1)

    # calculate extents of landmarks (rotated)
    cenrot_extents = np.amax(cenrots, axis=1) - np.amin(cenrots, axis=1)

    # NOTE: We add new columns by simple assignment here, before the DataFrame
    #       gets too big, to reduce index fragmentation.
    # add image name if it's missing
    if 'image_name' not in df.columns.values \
            and 'image-set' in df.columns.values \
            and 'filename' in df.columns.values:
        df['image_name'] = df[['image-set', 'filename']].agg('/'.join, axis=1)

    # add image dimensions
    df['boxratio'] = cenrot_extents[:, 0] / cenrot_extents[:, 1]
    df['boxsize'] = cenrot_extents[:, 0] * cenrot_extents[:, 1]
    df['width'] = widths
    df['height'] = heights
    df['face_width'] = cenrot_extents[:, 0]
    df['face_height'] = cenrot_extents[:, 1]
    df['cenrot_width'] = cenrot_widths
    df['cenrot_height'] = cenrot_heights

    # add interocular distance
    iods = (cenrots[:, 36, :] - cenrots[:, 45, :])
    df['interoc'] = np.sqrt(np.sum(np.power(iods, 2), axis=1))
    df['interoc_norm'] = np.sqrt(np.sum(np.power(iods / cenrot_extents, 2), axis=1))
    df['boxsize/interoc'] = df['boxsize'] / df['interoc']

    # ----- Estimated Angles
    # - estimated yaw (in radians)
    df['yaw'] = angles

    # - roll estimate
    # NOTE: this is based on the assumptions:
    #       * the head is a sphere from cheek to cheek
    #           (i.e. diameter is the max horizontal distance
    #           between 'cheeks' landmarks)
    #       * point 33 is on the surface of the head-sphere
    #       * point 33 is in the center of the face
    cheeks_x = cenrots[:, cheeks.ravel(), 0]
    min_x = np.min(cheeks_x, axis=1)
    radius = (np.max(cheeks_x, axis=1) - min_x) / 2
    mid_x = min_x + radius
    sins = np.clip((cenrots[:, nose_i, 0] - mid_x) / radius, -1, 1)
    df['roll'] = np.arcsin(sins)

    # - absolute values of estimated angles
    for col in ['yaw', 'roll']:
        df[f'{col}_abs'] = np.abs(df[col])

    # ----- Differences of Expected Symmetric Landmark Pairs
    # - raw
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(raws[:, [p1, p2], :], axis=1)),
            columns=[f'sym_diff-{dim}{p1}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    # - normalized
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(raws[:, [p1, p2], :], axis=1)) / extents,
            columns=[f'norm_sym_diff-{dim}{p1}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    # - normalized and rotated
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(cenrots[:, [p1, p2], :], axis=1)) / cenrot_extents,
            columns=[f'norm_cenrot_sym_diff-{dim}{p1}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    # ----- Landmark coordinates
    # - normalized
    #   (this is the distance from nose as a proportion of landmarks' extents)
    for i in range(raws.shape[1]):
        tmp_df = pd.DataFrame(
            data=(raws[:, i, :] - faces) / extents,
            columns=[f'norm-{dim}{i}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    # - centered/rotated
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=cenrots[:, i, :],
            columns=[f'cenrot-{dim}{i}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    # - centered, rotated and scaled per extent
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=(cenrots[:, i, :] - cenrots[:, nose_i, :]) / cenrot_extents,
            columns=[f'norm_cenrot-{dim}{i}' for dim in ['x', 'y']]
        )
        df = pd.concat([df, tmp_df], axis=1)

    return df
