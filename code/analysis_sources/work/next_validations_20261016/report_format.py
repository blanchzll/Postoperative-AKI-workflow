def markdown_table(rows, columns):
    # Create header
    header = '| ' + ' | '.join(columns) + ' |'
    
    # Create separator
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    
    # Create data rows
    data_rows = []
    for row in rows:
        row_data = []
        for col in columns:
            value = row.get(col, '')
            if value is None:
                value = ''
            value = str(value).replace('|', '\\|').replace('\n', ' ')
            row_data.append(value)
        data_rows.append('| ' + ' | '.join(row_data) + ' |')
    
    # Combine all parts
    table = [header, separator] + data_rows
    return '\n'.join(table)
