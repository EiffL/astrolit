# from astropy.table import Table, join
# from datasets import load_from_disk, concatenate_datasets
# import h5py
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from sparcl.client import SparclClient
import sys


def radec_string_to_degrees(ra_str, dec_str, ra_unit_formats, dec_unit_formats, st_obj):
    """convert from weird astronomer units to useful ones (degrees)"""

    ra_err_str = "The RA entered is not in the proper form: {:s}".format(
        ra_unit_formats
    )
    dec_err_str = "The Dec entered is not in the proper form: {:s}".format(
        dec_unit_formats
    )

    if ":" in ra_str:
        try:
            HH, MM, SS = [float(i) for i in ra_str.split(":")]
        except ValueError:
            st_obj.write(ra_err_str)
            sys.exit(ra_err_str)

        ra_str = 360.0 / 24 * (HH + MM / 60 + SS / 3600)

    if ":" in dec_str:
        try:
            DD, MM, SS = [float(i) for i in dec_str.split(":")]

        except ValueError:
            st_obj.write(dec_err_str)
            sys.exit(dec_err_str)
        dec_str = DD / abs(DD) * (abs(DD) + MM / 60 + SS / 3600)

    try:
        ra = float(ra_str)
    except ValueError:
        st_obj.write(ra_err_str)
        sys.exit(ra_err_str)

    try:
        dec = float(dec_str)
    except ValueError:
        st_obj.write(dec_err_str)
        sys.exit(dec_err_str)

    return ra, dec


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Angular separation between two points on a sphere.

    Parameters
    ----------
    ra1, dec1, ra2, dec2, : ra and dec in degrees

    Returns
    -------
    angular separation in degrees

    Notes
    -----
    see https://en.wikipedia.org/wiki/Great-circle_distance
    Adapted from Astropy https://github.com/astropy/astropy/blob/main/astropy/coordinates/angle_utilities.py. I am avoiding Astropy as it can be very slow
    """

    ra1 = np.radians(ra1)
    ra2 = np.radians(ra2)
    dec1 = np.radians(dec1)
    dec2 = np.radians(dec2)

    dsin_ra = np.sin(ra2 - ra1)
    dcos_ra = np.cos(ra2 - ra1)
    sin_dec1 = np.sin(dec1)
    sin_dec2 = np.sin(dec2)
    cos_dec1 = np.cos(dec1)
    cos_dec2 = np.cos(dec2)

    num1 = cos_dec2 * dsin_ra
    num2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * dcos_ra
    denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * dcos_ra

    return np.degrees(np.arctan2(np.hypot(num1, num2), denominator))


def random_object(provabgs_location):
    index_use_min = 2500

    ind_max = len(provabgs_location) - 1
    ind_random = 0
    while (ind_random < index_use_min) or (ind_random > ind_max):
        # ind_random = int(np.random.lognormal(10., 2.)) # strongly biased towards bright galaxies
        ind_random = int(
            np.random.lognormal(12.0, 3.0)
        )  # biased towards bright galaxies
    return provabgs_location[ind_random]


def search_catalogue(ra, dec, catalog, nnearest=1, far_distance_npix=10):

    sep = angular_separation(ra, dec, catalog["ra"], catalog["dec"])
    min_sep = 1e9
    query_min_sep = np.min(sep)
    query_index = np.argmin(sep)
    if query_min_sep < min_sep:
        return {
            "index": np.argmin(sep),
            "distance": sep[query_index],
            "ra": catalog["ra"][query_index],
            "dec": catalog["dec"][query_index],
            "targetid": catalog["targetid"][query_index],
            "min_sep": query_min_sep,
        }
    else:
        # no close galaxy found
        print("No galaxy found.")


def calculate_similarity(vector, embeddings):
    norm_vector = vector / (vector**2).sum(-1, keepdims=True)
    norm_embeddings = embeddings / (embeddings**2).sum(-1, keepdims=True)

    return (norm_embeddings @ norm_vector.T).squeeze()


def similarity_search(
    query_index,
    embeddings,
    nnearest=5,
    min_angular_separation=96,
):
    """
    Return indices and similarity scores to nearest nnearest data samples.
    First index returned is the query galaxy.

    Parameters
    ----------
    nnearest: int
        Number of most similar galaxies to return
    min_angular_separation: int
        Minimum angular seperation of galaxies in pixelsize. Anything below is thrown out
    similarity_inv: bool
        If True returns most similar, if False returns least similar
    """
    pixel_size = 0.262 / 3600  # arcsec to degrees

    query_object = embeddings[query_index]

    similarity_scores = calculate_similarity(query_object, embeddings)

    similarity_indices = np.argsort(-similarity_scores)[:nnearest]
    similarity_score = similarity_scores[similarity_indices]

    # TODO: this should be done at the dataset level
    # # now remove galaxies that are suspiciously close to each other on the sky
    # # which happens when individual galaxies in a cluster are included as separate sources in the catalogue
    # similarity_dict = load_from_catalogue_indices(
    #     include_extra_features=False, inds_load=similar_inds
    # )

    # # all vs all calculation
    # sep = angular_separation(
    #     similarity_dict["ra"][np.newaxis, ...],
    #     similarity_dict["dec"][np.newaxis, ...],
    #     similarity_dict["ra"][..., np.newaxis],
    #     similarity_dict["dec"][..., np.newaxis],
    # )

    # # compile indices of galaxies too close in angular coordinates
    # inds_del = set()
    # for i in range(sep.shape[0]):
    #     inds_cuti = set(
    #         np.where(sep[i, (i + 1) :] < min_angular_separation * pixel_size)[0]
    #         + (i + 1)
    #     )
    #     inds_del = inds_del | inds_cuti  # keep only unique indices

    # # remove duplicate galaxies from similarity arrays
    # inds_del = sorted(inds_del)
    # similar_inds = np.delete(similar_inds, inds_del)
    # similarity_score = np.delete(similarity_score, inds_del)

    return {"index": similarity_indices, "score": similarity_score}


RGB_SCALES = {
    "u": (2, 1.5),
    "g": (2, 6.0),
    "r": (1, 3.4),
    "i": (0, 1.0),
    "z": (0, 2.2),
}


# def decals_to_rgb(image, bands=["g", "r", "z"], scales=None, m=0.03, Q=20.0):
#     """Image processing function to convert DECaLS images to RGB.

#     Args:
#         image: torch.Tensor
#             The input image tensor with shape [batch, 3, npix, npix]
#     Returns:
#         torch.Tensor: The processed image tensor with shape [batch, 3, npix, npix, 3]
#     """
#     axes, scales = zip(*[RGB_SCALES[bands[i]] for i in range(len(bands))])
#     scales = [scales[i] for i in axes]
#     # Changing image shape to [batch_size, npix, npix, nchannel]
#     image = image.flip(-1)  # TODO: Figure out why flipping axis is necessary
#     scales = torch.tensor(scales, dtype=torch.float32).to(image.device)

#     I = torch.sum(torch.clamp(image * scales + m, min=0), dim=-1) / len(bands)

#     fI = torch.arcsinh(Q * I) / np.sqrt(Q)
#     I += (I == 0.0) * 1e-6

#     image = (image * scales + m) * (fI / I).unsqueeze(-1)
#     image = torch.clamp(image, 0, 1)

#     return image


def get_image_url_from_coordinates(ra: float, dec: float) -> str:
    return f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={float(ra)}&dec={float(dec)}&layer=ls-dr9-north&pixscale=0.262"


def get_spectrum_from_targets(client: SparclClient, targetids: list) -> np.ndarray:
    object_id = client.find(
        outfields=["sparcl_id"], constraints={"targetid": targetids}
    )
    retrieved_object = [client.retrieve([idx], include=["flux"]) for idx in object_id.ids]
    
    
    # bug in client: parallelization does not work (pickle file truncated)
    # retrieved_object = client.retrieve(object_id.ids, include=["flux"])
    return np.array([r[1]["flux"] for idx, r in enumerate(retrieved_object)])
