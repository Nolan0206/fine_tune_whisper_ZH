import zipfile
zip = zipfile.ZipFile("/mnt/common_voice2.zip")
zip.extractall("/mnt/common_voice")