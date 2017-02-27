import fs
from PIL import Image


def convert_dataset_to_png(original_dir, new_dir):
  root = fs.open_fs('/')
  original_dir = root.opendir(original_dir)
  new_dir = root.makedirs(new_dir, recreate=True)  # Ensure directory exists
  # Note: recreate=True suppresses DirectoryExists error;
  
  misc_files = []  # To collect and return miscellaneous non-image files
  error_files = []
  
  for f in original_dir.walk.files():
    if f.rfind('/') is not -1:
      if f[f.rfind('/')+1:].startswith('.'):  # Ignore hidden files
        continue
    elif f.startswith('.'): # Ignore hidden files
      continue
    elif f.endswith('.zip'):
      continue
    elif f.endswith('Thumbs.db'):  # Ignore Windows thumbnail files 
      continue
  
    original_file = original_dir.getsyspath(f)  # Get absolute path of file
    new_file = new_dir.getsyspath(f)  # Create the filepath in new directory
    
    # Capture the relative path to the directory containing the file
    parent_dir = f.rsplit('/', 1)[0]
    # Then create the corresponding directories in the new location
    new_dir.makedirs(parent_dir, recreate=True) 
    # Note: recreate=True surpresses DirectoryExists error; 
    # it doesn't delete existing files in those directories.
    
    #if f.endswith(('.jpg', '.bmp', '.tiff')):
    if f.endswith(('.bmp', '.tiff')):

      with Image.open(original_file) as image:
        new_file = new_file.rsplit('.', 1)[0]  # Remove filetype
        #print('SAVING %s' %  (new_file + '.png'))
        print('SAVING %s' %  (new_file + '.jpg'))

        try:
          image.save(new_file + '.jpg', 'JPEG')
        except ValueError:  # Some files seem to be partially corrupt.
          error_files.append(original_dir.getsyspath(f))
    else:
      #if not f.endswith('.png'):
      if not f.endswith('.jpg'):
        misc_files.append(f)
  return misc_files, error_files
