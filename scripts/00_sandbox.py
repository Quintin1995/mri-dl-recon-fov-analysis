import h5py
from xml.etree import ElementTree as ET
from pathlib import Path
import pydicom


def print_ismrmrd_header(path):
    with h5py.File(path, 'r') as df:
        # Assuming 'ismrmrd_header' is stored in XML format within the HDF5 file
        xml_header = df['ismrmrd_header'][()]
        
        # Decode bytes to string if necessary
        if isinstance(xml_header, bytes):
            xml_header = xml_header.decode('utf-8')
        
        # Parse the XML
        root = ET.fromstring(xml_header)
        
        # Pretty print the XML. Requires defusedxml if security is a concern.
        pretty_xml_as_string = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
        print(pretty_xml_as_string)
        x=4

if __name__ == "__main__":

    # Load the DICOM file
    dicom_file_path = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/data/0008_ANON8890538/dicoms/2022-05-25/tse2d1_25_1.3.12.2.1107.5.2.19.46133.2022052511231694378259504.0.0.0/sl_-0_404')
    ds = pydicom.dcmread(dicom_file_path)

    # Accessing the value of the private tag
    private_tag_value = ds.get((0x0051, 0x100A))

    # Extract just the string value, if the tag is present
    if private_tag_value is not None:
        # Access the .value attribute to get the raw value
        raw_value = private_tag_value.value
        print(f"Raw value of private tag (0051, 100A): {raw_value}")
    else:
        print("Tag (0051, 100A) not found in the DICOM file.")