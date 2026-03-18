export const metadata = { title: 'AI Policy Helper' };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div style={{maxWidth: 900, margin: '0 auto', padding: 24}}>
          {children}
        </div>
      </body>
    </html>
  );
}
