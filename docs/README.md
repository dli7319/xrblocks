# XR Blocks Documentation Site

Source code for https://xrblocks.github.io/docs/. \
This website is built using [Docusaurus](https://docusaurus.io/).

## Development

Start the development server for the documentation:

```bash
# In the xrblocks/docs folder:
npm start
```

For templates and samples in the documentation site to be loaded, a web server needs to be started in the `xrblocks` folder:

```bash
# In the xrblocks folder:
npm run build
http-server --cors
```

## Deployment

The documentation site is automatically deployed to https://xrblocks.github.io/docs/ whenever it is updated in the [google/xrblocks](https://github.com/google/xrblocks) repository.
