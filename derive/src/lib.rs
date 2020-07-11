use proc_macro2::TokenStream;
// use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{
    parse_macro_input, parse_quote, spanned::Spanned, Attribute, Data, DataStruct, DeriveInput,
    Error, Fields, GenericParam, Generics, Ident, Meta, NestedMeta,
};

#[derive(Debug, Clone)]
struct FieldAttr {
    pub clone_kind: CloneKind,
}

#[derive(Debug, Clone)]
enum CloneKind {
    Clone,
    Copy,
    None,
}

#[proc_macro_derive(TensorLike, attributes(tensor_like))]
pub fn derive_tensor_like(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let expanded = parse_tensor_like(input).unwrap_or_else(|err| err.to_compile_error());
    proc_macro::TokenStream::from(expanded)
}

fn parse_tensor_like(input: DeriveInput) -> Result<TokenStream, Error> {
    let name = input.ident;
    let generics = add_trait_bounds(input.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let f_to_device_impl = derive_f_to_device_impl(&input.data)?;
    let f_to_kind_impl = derive_f_to_kind_impl(&input.data)?;
    let shallow_clone_impl = derive_shallow_clone_impl(&input.data)?;

    let expanded = quote! {
        impl #impl_generics tch_tensor_like::TensorLike for #name #ty_generics #where_clause {
            fn f_to_device(&self, device: tch::Device) -> Result<Self, tch::TchError> {
                #f_to_device_impl
            }

            fn f_to_kind(&self, kind: tch::Kind) -> Result<Self, tch::TchError> {
                #f_to_kind_impl
            }

            fn shallow_clone(&self) -> Self {
                #shallow_clone_impl
            }
        }
    };

    Ok(expanded)
}

fn derive_impl<F>(data: &Data, transform: F) -> Result<TokenStream, Error>
where
    F: Fn(&Ident) -> TokenStream,
{
    let expanded = match data {
        Data::Struct(DataStruct {
            fields: Fields::Named(named_fields),
            ..
        }) => {
            let extract_idents = named_fields.named.iter().enumerate().map(|(index, field)| {
                let field_name = &field.ident;
                let proxy_name = format_ident!("_{}", index);
                quote_spanned! {
                    field.span() =>
                        #field_name: #proxy_name
                }
            });

            let recurse = named_fields
                .named
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let FieldAttr { clone_kind } = parse_field_attrs(&field.attrs)?;
                    let field_name = &field.ident;
                    let proxy_name = format_ident!("_{}", index);

                    let expanded_value = match clone_kind {
                        CloneKind::Clone => quote_spanned! {
                            field.span() =>
                                Clone::clone(#proxy_name)
                        },
                        CloneKind::Copy => quote_spanned! {
                            field.span() =>
                                *#proxy_name
                        },
                        CloneKind::None => transform(&proxy_name),
                    };

                    Ok(quote_spanned! {
                        field.span() =>
                            #field_name: #expanded_value
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;

            quote! {
                {
                    let Self {
                        #(#extract_idents),*
                    } = self;

                    Self {
                        #(#recurse),*
                    }
                }
            }
        }
        Data::Struct(DataStruct {
            fields: Fields::Unnamed(unnamed_fields),
            ..
        }) => {
            let extract_idents = unnamed_fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let proxy_name = format_ident!("_{}", index);
                    quote_spanned! {
                        field.span() =>
                            #proxy_name

                    }
                });

            let recurse = unnamed_fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    let FieldAttr { clone_kind } = parse_field_attrs(&field.attrs)?;
                    let ident = format_ident!("_{}", index);
                    let expanded_value = match clone_kind {
                        CloneKind::Clone => quote_spanned! {
                            field.span() =>
                                Clone::clone(#ident)
                        },
                        CloneKind::Copy => quote_spanned! {
                            field.span() =>
                                *#ident
                        },
                        CloneKind::None => transform(&ident),
                    };

                    Ok(quote_spanned! {
                        field.span() =>
                            #expanded_value
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;

            quote! {
                {
                    let Self (
                        #(#extract_idents),*
                    ) = self;

                    Self (
                        #(#recurse),*
                    )
                }
            }
        }
        Data::Struct(DataStruct {
            fields: Fields::Unit,
            ..
        }) => {
            quote! {
                Self
            }
        }
        Data::Enum(data) => {
            let recurse_variants = data
                .variants
                .iter()
                .map(|variant| {
                    let var_name = &variant.ident;

                    let expanded = match &variant.fields {
                        Fields::Named(fields) => {
                            let expanded_fields =
                                fields.named.iter().enumerate().map(|(index, field)| {
                                    let field_name = &field.ident;
                                    let proxy_name = format_ident!("_{}", index);

                                    quote_spanned! {
                                        field.span() =>
                                            #field_name: #proxy_name
                                    }
                                });

                            let recurse_fields = fields
                                .named
                                .iter()
                                .enumerate()
                                .map(|(index, field)| {
                                    let FieldAttr { clone_kind } = parse_field_attrs(&field.attrs)?;
                                    let field_name = &field.ident;
                                    let proxy_name = format_ident!("_{}", index);
                                    let expanded_value = match clone_kind {
                                        CloneKind::Clone => quote_spanned! {
                                            field.span() =>
                                                Clone::clone(#proxy_name)
                                        },
                                        CloneKind::Copy => quote_spanned! {
                                            field.span() =>
                                                *#proxy_name
                                        },
                                        CloneKind::None => transform(&proxy_name),
                                    };

                                    Ok(quote_spanned! {
                                        field.span() =>
                                            #field_name: #expanded_value
                                    })
                                })
                                .collect::<Result<Vec<_>, Error>>()?;

                            quote! {
                                Self::#var_name { #(#expanded_fields),* } => {
                                    Self::#var_name {
                                        #(#recurse_fields),*
                                    }
                                }
                            }
                        }
                        Fields::Unnamed(fields) => {
                            let expanded_fields =
                                fields.unnamed.iter().enumerate().map(|(index, field)| {
                                    let proxy_name = format_ident!("_{}", index);
                                    quote_spanned! {
                                        field.span() =>
                                            #proxy_name
                                    }
                                });

                            let recurse_fields = fields
                                .unnamed
                                .iter()
                                .enumerate()
                                .map(|(index, field)| {
                                    let FieldAttr { clone_kind } = parse_field_attrs(&field.attrs)?;
                                    let proxy_name = format_ident!("_{}", index);

                                    let expanded_value = match clone_kind {
                                        CloneKind::Clone => quote_spanned! {
                                            field.span() =>
                                                Clone::clone(#proxy_name)
                                        },
                                        CloneKind::Copy => quote_spanned! {
                                            field.span() =>
                                                *#proxy_name
                                        },
                                        CloneKind::None => transform(&proxy_name),
                                    };

                                    Ok(quote_spanned! {
                                        field.span() =>
                                            #expanded_value
                                    })
                                })
                                .collect::<Result<Vec<_>, Error>>()?;

                            quote! {
                                Self::#var_name ( #(#expanded_fields),* ) => {
                                    Self::#var_name (
                                        #(#recurse_fields),*
                                    )
                                }
                            }
                        }
                        Fields::Unit => {
                            quote! {
                                Self::#var_name => Self::#var_name
                            }
                        }
                    };

                    Ok(expanded)
                })
                .collect::<Result<Vec<_>, Error>>()?;

            quote! {
                match self {
                    #(#recurse_variants),*
                }
            }
        }
        Data::Union(_data) => quote! {
            compile_error!("union type is not supported")
        },
    };

    Ok(expanded)
}

fn parse_field_attrs(attrs: &[Attribute]) -> Result<FieldAttr, Error> {
    let mut is_clone = false;
    let mut is_copy = false;

    let metas_iter = attrs
        .iter()
        .filter(|attr| {
            attr.path
                .get_ident()
                .map(|ident| ident == &format_ident!("tensor_like"))
                .unwrap_or(false)
        })
        .map(|attr| (attr, attr.parse_meta()));

    for (attr, result) in metas_iter {
        let meta = match result {
            Ok(meta) => meta,
            Err(_) => return Err(Error::new(attr.span(), "expected #[tensor_like(...)]")),
        };

        let list = match meta {
            Meta::List(list) => list,
            _ => return Err(Error::new(attr.span(), "expected #[tensor_like(...)]")),
        };

        for nested in list.nested.iter() {
            let path = match nested {
                NestedMeta::Meta(Meta::Path(path)) => path,
                _ => return Err(Error::new(attr.span(), "expected #[tensor_like(...)]")),
            };

            let ident = path
                .get_ident()
                .ok_or_else(|| Error::new(attr.span(), "expected #[tensor_like(...)]"))?;

            match ident.to_string().as_str() {
                "clone" => {
                    is_clone = true;
                }
                "copy" => {
                    is_copy = true;
                }
                name @ _ => {
                    return Err(Error::new(
                        attr.span(),
                        format!(r#"unexpected attribute name "{}""#, name),
                    ))
                }
            }
        }
    }

    let clone_kind = if is_copy {
        CloneKind::Copy
    } else if is_clone {
        CloneKind::Clone
    } else {
        CloneKind::None
    };

    Ok(FieldAttr { clone_kind })
}

fn derive_f_to_device_impl(data: &Data) -> Result<TokenStream, Error> {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                tch_tensor_like::TensorLike::f_to_device(#ident, device)?
        }
    })?;

    Ok(quote! {
        Ok(#expanded)
    })
}

fn derive_f_to_kind_impl(data: &Data) -> Result<TokenStream, Error> {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                tch_tensor_like::TensorLike::f_to_kind(#ident, kind)?
        }
    })?;

    Ok(quote! {
        Ok(#expanded)
    })
}

fn derive_shallow_clone_impl(data: &Data) -> Result<TokenStream, Error> {
    let expanded = derive_impl(data, |ident| {
        quote_spanned! {
            ident.span() =>
                tch_tensor_like::TensorLike::shallow_clone(#ident)
        }
    })?;

    Ok(expanded)
}

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param {
            type_param
                .bounds
                .push(parse_quote!(tch_tensor_like::TensorLike));
        }
    }
    generics
}
