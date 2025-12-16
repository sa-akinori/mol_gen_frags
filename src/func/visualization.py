from copy import deepcopy
import xlsxwriter 
import pandas as pd
import numpy as np
import tempfile
import os
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage

def AddHeadFormat(workbook):
    """
    Add a style to the workbook that is used in the header of the table
    """
    format = workbook.add_format(dict(bold=True, align='center', valign='vcenter', size=12))
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()
    return format

def AddDataFormat(workbook):
    """
    Add a style to the workbook data place 
    """
    format = workbook.add_format(dict(bold=False, align='center', valign='vcenter', size=15))
    format.set_text_wrap()
    format.set_bg_color('white')
    format.set_border_color('black')
    format.set_border()
    
    return format

def writeImageToFile(mol, ofile, width=300, height=300):
    """
    write molecule image to a file with formatting options
    """       
    img = MolToImage(mol, (width, height))
    img.save(ofile, bitmap_format='png')
    
def WriteDataFrameSmilesToXls(pd_table, smiles_colnames, out_filename,  smirks_colnames=None, max_mols=10000, 
                            retain_smiles_col=False, use_romol=False):
    """
    Write panads DataFrame containing smiles as molcular image
	rdkit version...

    input:
    ------
    pd_table:
    smiles_colnames: must set the smiles column names where smiles are converted to images
    max_mol: For avoid generating too much data 
    out_filename: output file name 
    smirks_colname: reaction smiles (smirks) column name which is decomposed to left and right parts to visualization
    retain_smiles_col: (retaining SMIELS columns or mot )
    output:
    ------
    None: 

    """
    if len(pd_table) == 0:
        print('No data found in the table. Do nothing.')
        return 
    if isinstance(smiles_colnames, str):
        smiles_colnames = [smiles_colnames]
    
    if smiles_colnames is None:
        smiles_colnames = ['']

    if use_romol:
        pd_table[smiles_colnames] = pd_table[smiles_colnames].apply(lambda x: Chem.MolToSmiles(x) if x is not None else '') 
    
    if retain_smiles_col:
        pd_smiles = pd_table[smiles_colnames].copy()
        pd_smiles.columns = ['{}_SMI'.format(s) for s in smiles_colnames]
        pd_table = pd.concat([pd_table, pd_smiles], axis=1)

    if smirks_colnames is not None:
        if isinstance(smirks_colnames, str):
            smirks_colnames = [smirks_colnames]
        for smirks_col in smirks_colnames:
            lname, midname, rname   = f'left_{smirks_col}', f'middle_{smirks_col}', f'right_{smirks_col}'
            pd_table[lname]         = pd_table[smirks_col].str.split('>').str[0]
            pd_table[midname]       = pd_table[smirks_col].str.split('>').str[1]
            pd_table[rname]         = pd_table[smirks_col].str.split('>').str[2]
            
            # check the middle part (if no condition for all the smirks, remove it)
            if (pd_table[midname]=='').all():
                del pd_table[midname]
                smirks_names = [lname, rname]
            else:
                smirks_names = [lname, midname, rname]
            smiles_colnames.extend(smirks_names)

    # if the column contain objects then it convers to string
    array_columns = [col for col in pd_table.columns if isinstance(pd_table[col].iloc[0], np.ndarray)]
    pd_table[array_columns] = pd_table[array_columns].apply(lambda x: str(x))

    # set up depiction option
    width, height = 250, 250
   
    if not isinstance(pd_table, pd.DataFrame):
        raise ValueError("pd_table must be pandas DataFrame")
    
    if len(pd_table) > max_mols:
        raise ValueError('maximum number of rows is set to %d but input %d' %(max_mols, len(pd_table)))

    workbook = xlsxwriter.Workbook(out_filename)
    worksheet = workbook.add_worksheet()

    # Set header to a workbook
    headformat = AddHeadFormat(workbook)
    dataformat = AddDataFormat(workbook)

    # Estimate the width of columns
    maxwidths = dict()
    
    if not pd_table.index.name:
        pd_table.index.name = 'index'
    
    for column in pd_table:
        if column in smiles_colnames: # for structure column
            maxwidths[column] = width *0.15 # I do not know why this works
        else:
            if pd_table[column].dtype == list: # list to str
                pd_table[column] = pd_table[column].apply(str)
            l_txt = pd_table[column].apply(lambda x: len(str(x)))
            
            l_len = np.max(l_txt)
            l_len = max(l_len, len(str(column)))
            maxwidths[column] = l_len * 1.2  # misterious scaling
    
    # Generate header (including index part)
    row, col = 0, 0
    worksheet.set_row(row, None, headformat)
    worksheet.set_column(col, col, len(str(pd_table.index.name)))
    worksheet.write(row, col, pd_table.index.name)
    
    for colname in pd_table:
        col +=1
        worksheet.set_column(col, col, maxwidths[colname])
        worksheet.write(row, col, colname)

    # temporary folder for storing figs
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Generate the data
        for idx, val in pd_table.iterrows():
            row += 1
            worksheet.set_row(row, height * 0.75, dataformat)

            col = 0
            worksheet.write(row, col, idx)
            
            # contents 
            for cname, item in val.items():
                col += 1
                if cname in smiles_colnames:
                    fname = os.path.join(tmp_dir, '%d_%d.png' %(row, col)) 
                    if isinstance(item, str):
                        mol = Chem.MolFromSmiles(item)
                    else:
                        mol = item
                    if mol is not None:
                        writeImageToFile(mol, fname, int(width*0.9), int(height*0.9))
                        worksheet.insert_image(row, col, fname, dict(object_position=1, x_offset=1, y_offset=1))
                else:
                    try:
                        worksheet.write(row, col, item)
                    except: 
                        continue
        workbook.close()